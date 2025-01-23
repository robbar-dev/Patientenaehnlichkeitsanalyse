import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import datetime

from torch.utils.data import DataLoader

# Füge Dein Projektverzeichnis hinzu
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 1) ResNet18-Feature Extractor
from model.base_cnn import BaseCNN

# 2) Aggregatoren
from model.mil_aggregator import AttentionMILAggregator
from model.max_pooling import MaxPoolingAggregator
from model.mean_pooling import MeanPoolingAggregator

# 3) TripletSampler
from training.triplet_sampler import TripletSampler

# 4) SinglePatientDataset mit Filtern & Normalisierung
from training.data_loader import SinglePatientDataset

# 5) Evaluate-Funktionen
from evaluation.metrics import evaluate_model

import matplotlib
matplotlib.use('Agg')  # kein GUI-Fenster
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class TripletTrainer(nn.Module):
    """
    Ein erweiterter Trainer (nn.Module), der:
      - base_cnn (ResNet18 oder ResNet50)
      - mil_agg (AttentionMIL, MaxPooling, MeanPooling)
      - TripletMarginLoss
      - Adam-Optimizer
      - (Optional) Scheduler

    Er enthält:
      - train_loop(...), train_with_val(...)
      - compute_patient_embedding(...) => Validation-Embedding
      - _forward_patient(...) => Training
      - Visualisierungen: Loss-Kurve, t-SNE/PCA, etc.
    """

    def __init__(
        self,
        aggregator_name,
        df,
        data_root,
        device='cuda',
        lr=1e-3,
        margin=1.0,
        roi_size=(96, 96, 3),
        overlap=(10, 10, 1),
        pretrained=False,
        attention_hidden_dim=128,
        dropout=0.2,
        weight_decay=1e-5,
        freeze_blocks=None,
        # Neu: Parameter für SinglePatientDataset
        skip_slices=False,
        skip_factor=2,
        filter_empty_patches=True,
        min_nonzero_fraction=0.01,
        do_patch_minmax=True
    ):
        """
        Args:
            aggregator_name: 'mil','max','mean'
            df: Pandas DataFrame mit [pid, study_yr, combination]
            data_root: Pfad zu den NIfTI .nii.gz
            device: 'cuda'/'cpu'
            lr: Lernrate
            margin: Triplet-Margin
            roi_size, overlap: Patch-Extraktion
            pretrained: ob ResNetX auf ImageNet-Weights basiert
            attention_hidden_dim: Hidden-Layer im Attention-MIL
            dropout: Dropout-Rate
            weight_decay: L2-Reg
            freeze_blocks: z.B. [0,1] => layer1+layer2 freezen
            skip_slices, skip_factor: Ob wir volume[..., ::2] etc. benutzen
            filter_empty_patches: ob wir Patches, die ~leer sind, wegwerfen
            min_nonzero_fraction: float, Schwelle an Nonzero
            do_patch_minmax: pro Patch minmax normalisieren
        """
        super().__init__()
        self.df = df
        self.data_root = data_root
        self.device = device
        self.roi_size = roi_size
        self.overlap = overlap

        # -> Parameter an SinglePatientDataset
        self.skip_slices = skip_slices
        self.skip_factor = skip_factor
        self.filter_empty_patches = filter_empty_patches
        self.min_nonzero_fraction = min_nonzero_fraction
        self.do_patch_minmax = do_patch_minmax

        # (A) CNN-Backbone
        self.base_cnn = BaseCNN(
            model_name='resnet18',
            pretrained=pretrained,
            freeze_blocks=freeze_blocks
        ).to(device)

        # (B) Aggregation
        if aggregator_name == "mil":
            self.mil_agg = AttentionMILAggregator(
                in_dim=512,
                hidden_dim=attention_hidden_dim,
                dropout=dropout
            ).to(device)
        elif aggregator_name == "max":
            self.mil_agg = MaxPoolingAggregator().to(device)
        elif aggregator_name == "mean":
            self.mil_agg = MeanPoolingAggregator().to(device)
        else:
            raise ValueError(f"Unbekannter Aggregator: {aggregator_name}")

        # (C) TripletLoss
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

        # (D) Optimizer
        self.optimizer = optim.Adam(
            list(self.base_cnn.parameters()) + list(self.mil_agg.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        # (E) Optional Scheduler
        self.scheduler = None

        # Tracking
        self.best_val_map = 0.0
        self.best_val_epoch = -1

        # Epochen-Loss und Metrics
        self.epoch_losses = []
        self.metric_history = {
            "precision": [],
            "recall": [],
            "mAP": []
        }

        logging.info(
            f"[TripletTrainer] aggregator={aggregator_name}, lr={lr}, margin={margin}, "
            f"dropout={dropout}, weight_decay={weight_decay}, freeze_blocks={freeze_blocks}, "
            f"skip_slices={skip_slices}, skip_factor={skip_factor}, "
            f"filter_empty_patches={filter_empty_patches}, min_nonzero_frac={min_nonzero_fraction}, "
            f"do_patch_minmax={do_patch_minmax}"
        )


    # --------------------------------------------------------------------------
    # compute_patient_embedding (Val/Test) => SinglePatientDataset
    # --------------------------------------------------------------------------
    def compute_patient_embedding(self, pid, study_yr):
        """
        Lädt alle Patches => CNN => MIL => (1,512)
        (keine Grad-Berechnung)
        """
        ds = SinglePatientDataset(
            data_root=self.data_root,
            pid=pid,
            study_yr=study_yr,
            roi_size=self.roi_size,
            overlap=self.overlap,
            skip_slices=self.skip_slices,
            skip_factor=self.skip_factor,
            filter_empty_patches=self.filter_empty_patches,
            min_nonzero_fraction=self.min_nonzero_fraction,
            do_patch_minmax=self.do_patch_minmax
        )
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        all_embs = []

        self.base_cnn.eval()
        self.mil_agg.eval()
        with torch.no_grad():
            for patch_t in loader:
                patch_t = patch_t.to(self.device)
                emb = self.base_cnn(patch_t)  # => (B,512)
                all_embs.append(emb)

        if len(all_embs) == 0:
            return torch.zeros((1,512), device=self.device)

        patch_embs = torch.cat(all_embs, dim=0)  # => (N,512)
        patient_emb = self.mil_agg(patch_embs)   # => (1,512)
        return patient_emb

    # --------------------------------------------------------------------------
    # _forward_patient (Training) => CNN => MIL => (1,512) mit Grad
    # --------------------------------------------------------------------------
    def _forward_patient(self, pid, study_yr):
        ds = SinglePatientDataset(
            data_root=self.data_root,
            pid=pid,
            study_yr=study_yr,
            roi_size=self.roi_size,
            overlap=self.overlap,
            skip_slices=self.skip_slices,
            skip_factor=self.skip_factor,
            filter_empty_patches=self.filter_empty_patches,
            min_nonzero_fraction=self.min_nonzero_fraction,
            do_patch_minmax=self.do_patch_minmax
        )
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        patch_embs = []

        self.base_cnn.train()
        self.mil_agg.train()
        for patch_t in loader:
            patch_t = patch_t.to(self.device)
            emb = self.base_cnn(patch_t)  # => (B,512)
            patch_embs.append(emb)

        if len(patch_embs) == 0:
            # kein Patch =>return dummy
            return torch.zeros((1,512), device=self.device, requires_grad=True)

        patch_embs = torch.cat(patch_embs, dim=0)  # =>(N,512)
        patient_emb = self.mil_agg(patch_embs)     # =>(1,512)
        return patient_emb

    # --------------------------------------------------------------------------
    # train_one_epoch => Sampler => TripletLoss
    # --------------------------------------------------------------------------
    def train_one_epoch(self, sampler):
        total_loss = 0.0
        steps = 0
        for step, (anchor_info, pos_info, neg_info) in enumerate(sampler):
            # anchor_pid, anchor_sy, ...
            a_pid, a_sy = anchor_info['pid'], anchor_info['study_yr']
            p_pid, p_sy = pos_info['pid'], pos_info['study_yr']
            n_pid, n_sy = neg_info['pid'], neg_info['study_yr']

            anchor_emb = self._forward_patient(a_pid, a_sy)
            pos_emb    = self._forward_patient(p_pid, p_sy)
            neg_emb    = self._forward_patient(n_pid, n_sy)

            loss = self.triplet_loss_fn(anchor_emb, pos_emb, neg_emb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

            if step%250 == 0:
                logging.info(f"[Step {step}] Triplet Loss = {loss.item():.4f}")

        avg_loss = total_loss / steps if steps>0 else 0.0
        logging.info(f"Epoch Loss = {avg_loss:.4f}")
        self.epoch_losses.append(avg_loss)

    # --------------------------------------------------------------------------
    # train_loop => N Epochen
    # --------------------------------------------------------------------------
    def train_loop(self, num_epochs=2, num_triplets=100):
        sampler = TripletSampler(
            df=self.df,
            num_triplets=num_triplets,
            shuffle=True,
            top_k_negatives=3
        )
        for epoch in range(1, num_epochs+1):
            self.train_one_epoch(sampler)

            # optional scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                logging.info(f"Nach Epoche {epoch}, LR={current_lr}")

            sampler.reset_epoch()

    # --------------------------------------------------------------------------
    # train_with_val => (N Epochen) => pro Epoche train_loop(1)
    #                   => Evaluate => track best => Visualisierung
    # --------------------------------------------------------------------------
    def train_with_val(
        self,
        epochs,
        num_triplets,
        val_csv,
        data_root_val,
        K=10,
        distance_metric='euclidean',
        visualize_every=5,
        visualize_method='tsne',
        output_dir='plots'
    ):
        self.best_val_map = 0.0
        self.best_val_epoch = -1
        self.epoch_losses = []
        self.metric_history = {
            "precision": [],
            "recall": [],
            "mAP": []
        }

        df_val = pd.read_csv(val_csv)

        for epoch in range(1, epochs+1):
            logging.info(f"=== EPOCH {epoch}/{epochs} ===")
            # 1 Epoche train
            self.train_loop(num_epochs=1, num_triplets=num_triplets)

            # Evaluate => val
            val_metrics = evaluate_model(
                trainer=self,
                data_csv=val_csv,
                data_root=data_root_val,
                K=K,
                distance_metric=distance_metric,
                device=self.device
            )
            precisionK = val_metrics["precision@K"]
            recallK    = val_metrics["recall@K"]
            map_val    = val_metrics["mAP"]

            logging.info(f"Val-Epoch={epoch}: precision={precisionK:.4f}, recall={recallK:.4f}, mAP={map_val:.4f}")

            self.metric_history["precision"].append(precisionK)
            self.metric_history["recall"].append(recallK)
            self.metric_history["mAP"].append(map_val)

            # best tracking
            if map_val>self.best_val_map:
                self.best_val_map = map_val
                self.best_val_epoch = epoch
                logging.info(f"=> New Best mAP={map_val:.4f} @ epoch={epoch}")

            # Visualisierung
            if epoch % visualize_every==0:
                self.visualize_embeddings(
                    df=df_val,
                    data_root=data_root_val,
                    method=visualize_method,
                    epoch=epoch,
                    output_dir=output_dir
                )

        self.plot_loss_curve(output_dir=output_dir)
        self.plot_metric_curves(output_dir=output_dir)

        return (self.best_val_map, self.best_val_epoch)

    # --------------------------------------------------------------------------
    # visualize_embeddings => t-SNE / PCA
    # --------------------------------------------------------------------------
    def visualize_embeddings(
        self,
        df,
        data_root,
        method='tsne',
        epoch=0,
        output_dir='plots'
    ):
        logging.info(f"Visualisiere Embeddings => {method.upper()}, EPOCH={epoch}")
        embeddings_list = []
        combos = []
        for i, row in df.iterrows():
            pid = row['pid']
            study_yr = row['study_yr']
            combo = row['combination']

            emb = self.compute_patient_embedding(pid, study_yr)  # => (1,512)
            emb_np = emb.squeeze(0).cpu().numpy()
            embeddings_list.append(emb_np)
            combos.append(combo)

        embeddings_arr = np.array(embeddings_list)  # (N,512)
        if method.lower()=='tsne':
            from sklearn.manifold import TSNE
            projector = TSNE(n_components=2, random_state=42)
        else:
            from sklearn.decomposition import PCA
            projector = PCA(n_components=2, random_state=42)

        coords_2d = projector.fit_transform(embeddings_arr)  # => (N,2)

        plt.figure(figsize=(8,6))
        unique_combos = sorted(list(set(combos)))
        for c in unique_combos:
            idxs = [ix for ix,val in enumerate(combos) if val==c]
            plt.scatter(coords_2d[idxs,0], coords_2d[idxs,1], label=str(c), alpha=0.6)

        plt.title(f"{method.upper()} Embeddings - EPOCH={epoch}")
        plt.legend(loc='best')

        aggregator_name = getattr(self.mil_agg, '__class__').__name__
        timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fname = os.path.join(
            output_dir,
            f"Emb_{method}_{aggregator_name}_epoch{epoch:03d}_{timestamp}.png"
        )
        plt.savefig(fname, dpi=150)
        plt.close()
        logging.info(f"Saved embedding plot: {fname}")

    # --------------------------------------------------------------------------
    # plot_loss_curve => Speichert die Loss-Kurve
    # --------------------------------------------------------------------------
    def plot_loss_curve(self, output_dir='plots'):
        if not self.epoch_losses:
            logging.info("Keine epoch_losses => skip plot_loss_curve.")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        x_vals = list(range(1, len(self.epoch_losses)+1))
        plt.figure(figsize=(8,6))
        plt.plot(x_vals, self.epoch_losses, marker='o', color='navy', label='Train Loss')
        plt.title("Train Loss pro Epoche")
        plt.xlabel("Epoch")
        plt.ylabel("Triplet Loss")
        plt.legend()

        aggregator_name = getattr(self.mil_agg, '__class__').__name__
        timestr = datetime.datetime.now().strftime("%m%d-%H%M")
        outname = os.path.join(output_dir, f"loss_vs_epoch_{aggregator_name}_{timestr}.png")
        plt.savefig(outname, dpi=150)
        plt.close()
        logging.info(f"Saved loss curve: {outname}")

    # --------------------------------------------------------------------------
    # plot_metric_curves => Precision, Recall, mAP vs Epoche
    # --------------------------------------------------------------------------
    def plot_metric_curves(self, output_dir='plots'):
        if len(self.metric_history["mAP"])==0:
            logging.info("Keine Metriken => skip plot_metric_curves.")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        x_vals = list(range(1, len(self.metric_history["mAP"])+1))

        plt.figure(figsize=(8,6))
        plt.plot(x_vals, self.metric_history["precision"], 'g-o', label='Precision@K')
        plt.plot(x_vals, self.metric_history["recall"], 'm-s', label='Recall@K')
        plt.plot(x_vals, self.metric_history["mAP"], 'r-^', label='mAP')
        plt.ylim(0,1)
        plt.title("Precision@K, Recall@K, mAP vs Epoche")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()

        aggregator_name = getattr(self.mil_agg, '__class__').__name__
        timestr = datetime.datetime.now().strftime("%m%d-%H%M")
        outname = os.path.join(output_dir, f"metrics_vs_epoch_{aggregator_name}_{timestr}.png")
        plt.savefig(outname, dpi=150)
        plt.close()
        logging.info(f"Saved metric curves: {outname}")

    # --------------------------------------------------------------------------
    # save_checkpoint / load_checkpoint
    # --------------------------------------------------------------------------
    def save_checkpoint(self, path):
        ckpt = {
            'base_cnn': self.base_cnn.state_dict(),
            'mil_agg': self.mil_agg.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if self.scheduler is not None:
            ckpt['scheduler'] = self.scheduler.state_dict()
        torch.save(ckpt, path)
        logging.info(f"Checkpoint saved => {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.base_cnn.load_state_dict(ckpt['base_cnn'])
        self.mil_agg.load_state_dict(ckpt['mil_agg'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

        if self.scheduler is not None and 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])

        logging.info(f"Checkpoint loaded from {path}")


if __name__=="__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Beispiel
    # data_csv = r"C:\Users\xxx\train.csv"
    # data_root = r"D:\my_nifti"
    # df = pd.read_csv(data_csv)
    # trainer = TripletTrainer(
    #     aggregator_name="mil",
    #     df=df,
    #     data_root=data_root,
    #     device='cuda',
    #     lr=1e-4,
    #     margin=1.0,
    #     roi_size=(96,96,3),
    #     overlap=(10,10,1),
    #     pretrained=False,
    #     attention_hidden_dim=128,
    #     dropout=0.2,
    #     weight_decay=1e-5,
    #     freeze_blocks=None,
    #     skip_slices=False,
    #     skip_factor=2,
    #     filter_empty_patches=True,
    #     min_nonzero_fraction=0.01,
    #     do_patch_minmax=True
    # )
    #
    # best_map, best_epoch = trainer.train_with_val(
    #     epochs=10,
    #     num_triplets=500,
    #     val_csv=r"C:\...\val.csv",
    #     data_root_val=data_root,
    #     K=5,
    #     distance_metric='euclidean',
    #     visualize_every=2,
    #     visualize_method='pca',
    #     output_dir='plots'
    # )
    # logging.info(f"Fertig. Best mAP={best_map:.4f} in Epoche={best_epoch}")
