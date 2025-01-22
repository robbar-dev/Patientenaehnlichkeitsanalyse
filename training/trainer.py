import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import datetime

from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.base_cnn import BaseCNN
from model.mil_aggregator import AttentionMILAggregator
from model.max_pooling import MaxPoolingAggregator
from model.mean_pooling import MeanPoolingAggregator
from training.triplet_sampler import TripletSampler
from training.data_loader import SinglePatientDataset

from evaluation.metrics import evaluate_model

import matplotlib
matplotlib.use('Agg')  # damit kein GUI-Fenster geöffnet wird (z. B. auf Server)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class TripletTrainer(nn.Module):
    """
    Ein erweiterter Trainer, der nn.Module erbt, um state_dict() und load_state_dict()
    nativ verwenden zu können. Enthält:
    - base_cnn (ResNet18) mit optionalem Teil-Freeze
    - mil_agg (AttentionMIL, MaxPooling, MeanPooling)
    - triplet_loss
    - optimizer (+ optional scheduler)
    - Methoden: train_loop, train_with_val, compute_patient_embedding, ...
    - Visualisierungen:
      -> Loss-Kurve
      -> t-SNE/PCA-Einbettungen (alle N Epochen)
      -> Precision@K, Recall@K, mAP-Kurven
    - save_checkpoint / load_checkpoint

    Ablauf:
      1) train_with_val(...) läuft Epochen durch
      2) pro Epoche => train_loop(num_epochs=1)
      3) evaluate_model(...) erfasst (Precision@K, Recall@K, mAP)
      4) Speichern der Metriken in self.metric_history
      5) Am Ende Plotten:
         - self.plot_loss_curve()
         - self.plot_metric_curves() (Precision@K, Recall@K, mAP)
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
        freeze_blocks=None
    ):
        """
        Args:
            aggregator_name: 'mil', 'max', oder 'mean'
            df: Pandas DataFrame [pid, study_yr, combination]
            data_root: Pfad zu .nii.gz
            device: 'cuda'/'cpu'
            lr, margin: Hyperparams
            roi_size, overlap: Patch-Extraktion
            pretrained: ob ResNet18 auf ImageNet-Weights basiert
            attention_hidden_dim: Hidden für Gated-Attention
            dropout: Dropout-Rate im Aggregator
            weight_decay: L2-Regularization
            freeze_blocks: None oder [0,1,2...] um ResNet-Blocks einzufrieren
        """
        super().__init__()
        self.df = df
        self.data_root = data_root
        self.device = device
        self.roi_size = roi_size
        self.overlap = overlap

        # 1) CNN-Backbone
        self.base_cnn = BaseCNN(
            model_name='resnet50',
            pretrained=pretrained,
            freeze_blocks=freeze_blocks
        ).to(device)

        # 2) Aggregator
        if aggregator_name == "mil":
            self.mil_agg = AttentionMILAggregator(
                in_dim=2048,
                hidden_dim=attention_hidden_dim,
                dropout=dropout
            ).to(device)
        elif aggregator_name == "max":
            self.mil_agg = MaxPoolingAggregator().to(device)
        elif aggregator_name == "mean":
            self.mil_agg = MeanPoolingAggregator().to(device)
        else:
            raise ValueError(f"Unbekannter Aggregator: {aggregator_name}")

        # 3) TripletLoss
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

        # 4) Optimizer
        self.optimizer = optim.Adam(
            list(self.base_cnn.parameters()) + list(self.mil_agg.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        # 5) Optional Scheduler
        self.scheduler = None

        # Best Val-Tracking
        self.best_val_map = 0.0
        self.best_val_epoch = -1

        # Epochen-Loss Liste
        self.epoch_losses = []

        # NEU: Metrik-Verlauf (Epoche => (Precision@K, Recall@K, mAP))
        self.metric_history = {
            "precision": [],
            "recall": [],
            "mAP": []
        }

        logging.info(
            f"[TripletTrainer] aggregator={aggregator_name}, lr={lr}, margin={margin}, "
            f"dropout={dropout}, weight_decay={weight_decay}, freeze_blocks={freeze_blocks}"
        )

    # --------------------------------------
    # compute_patient_embedding (Val/Test)
    # --------------------------------------
    def compute_patient_embedding(self, pid, study_yr):
        ds = SinglePatientDataset(
            data_root=self.data_root,
            pid=pid,
            study_yr=study_yr,
            roi_size=self.roi_size,
            overlap=self.overlap
        )
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        all_embs = []
        self.base_cnn.eval()
        self.mil_agg.eval()

        with torch.no_grad():
            for patch_t in loader:
                patch_t = patch_t.to(self.device)
                emb = self.base_cnn(patch_t)  # => (B,2048)
                all_embs.append(emb)

        if len(all_embs) == 0:
            return torch.zeros((1,2048), device=self.device)

        patch_embs = torch.cat(all_embs, dim=0)  # => (N,2048)
        patient_emb = self.mil_agg(patch_embs)   # => (1,2048)
        return patient_emb

    # --------------------------------------
    # _forward_patient (Training)
    # --------------------------------------
    def _forward_patient(self, pid, study_yr):
        ds = SinglePatientDataset(
            data_root=self.data_root,
            pid=pid,
            study_yr=study_yr,
            roi_size=self.roi_size,
            overlap=self.overlap
        )
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        patch_embs = []
        self.base_cnn.train()
        self.mil_agg.train()

        for patch_t in loader:
            patch_t = patch_t.to(self.device)
            emb = self.base_cnn(patch_t)  # => (B,2048)
            patch_embs.append(emb)

        if len(patch_embs) == 0:
            return torch.zeros((1,2048), device=self.device, requires_grad=True)

        patch_embs = torch.cat(patch_embs, dim=0)
        patient_emb = self.mil_agg(patch_embs)  # => (1,2048)
        return patient_emb

    # --------------------------------------
    # train_one_epoch => Triplet-Loss
    # --------------------------------------
    def train_one_epoch(self, sampler):
        total_loss = 0.0
        steps = 0
        for step, (anchor_info, pos_info, neg_info) in enumerate(sampler):
            anchor_pid, anchor_sy = anchor_info['pid'], anchor_info['study_yr']
            pos_pid, pos_sy       = pos_info['pid'], pos_info['study_yr']
            neg_pid, neg_sy       = neg_info['pid'], neg_info['study_yr']

            anchor_emb = self._forward_patient(anchor_pid, anchor_sy)
            pos_emb    = self._forward_patient(pos_pid, pos_sy)
            neg_emb    = self._forward_patient(neg_pid, neg_sy)

            loss = self.triplet_loss_fn(anchor_emb, pos_emb, neg_emb)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

            if step % 250 == 0:
                logging.info(f"[Step {step}] Triplet Loss = {loss.item():.4f}")

        avg_loss = total_loss / steps if steps > 0 else 0.0
        logging.info(f"Epoch Loss = {avg_loss:.4f}")
        self.epoch_losses.append(avg_loss)

    # --------------------------------------
    # train_loop => Schleife (N Epochen)
    # --------------------------------------
    def train_loop(self, num_epochs=2, num_triplets=100):
        sampler = TripletSampler(
            df=self.df,
            num_triplets=num_triplets,
            shuffle=True,
            top_k_negatives=3
        )
        for epoch in range(1, num_epochs + 1):
            self.train_one_epoch(sampler)
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                logging.info(f"Nach Epoche {epoch}, LR={current_lr}")
            sampler.reset_epoch()

    # --------------------------------------------------------------------------
    # train_with_val => Train + Val + Visualisierung
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
        """
        Führt ein Training + Validierung durch.
         - epochs Mal
         - pro Epoche => train_loop(num_epochs=1)
         - evaluate_model => (precision, recall, mAP)
         - track best_val_map
         - speichert Epochen-Loss => Plot am Ende
         - optional: visualisiert Embeddings alle visualize_every Epochen
         - speichert Metric-Kurven (Precision@K, Recall@K, mAP) am Ende
        """
        self.best_val_map = 0.0
        self.best_val_epoch = -1
        self.epoch_losses = []
        self.metric_history = {
            "precision": [],
            "recall": [],
            "mAP": []
        }

        df_val = pd.read_csv(val_csv)

        for epoch in range(1, epochs + 1):
            logging.info(f"=== EPOCH {epoch}/{epochs} ===")
            # 1 Epoche train
            self.train_loop(num_epochs=1, num_triplets=num_triplets)

            # Evaluate auf Val
            val_metrics = evaluate_model(
                trainer=self,
                data_csv=val_csv,
                data_root=data_root_val,
                K=K,
                distance_metric=distance_metric,
                device=self.device
            )
            current_precision = val_metrics['precision@K']
            current_recall    = val_metrics['recall@K']
            current_map       = val_metrics['mAP']

            logging.info(
                f"Val-Epoch={epoch}: precision={current_precision:.4f}, "
                f"recall={current_recall:.4f}, mAP={current_map:.4f}"
            )

            # Speichere Metriken
            self.metric_history["precision"].append(current_precision)
            self.metric_history["recall"].append(current_recall)
            self.metric_history["mAP"].append(current_map)

            # Track Best
            if current_map > self.best_val_map:
                self.best_val_map = current_map
                self.best_val_epoch = epoch
                logging.info(f"=> New Best mAP={self.best_val_map:.4f} in Epoche {epoch}")

            # Visualisierung => Embeddings
            if epoch % visualize_every == 0:
                self.visualize_embeddings(
                    df=df_val,
                    data_root=data_root_val,
                    method=visualize_method,
                    epoch=epoch,
                    output_dir=output_dir
                )

        # Am Ende => Plots generieren
        self.plot_loss_curve(output_dir=output_dir)
        self.plot_metric_curves(output_dir=output_dir)

        return (self.best_val_map, self.best_val_epoch)

    # --------------------------------------------------------------------------
    # visualisieren der Embeddings => t-SNE oder PCA
    # --------------------------------------------------------------------------
    def visualize_embeddings(self,
                             df,
                             data_root,
                             method='tsne',
                             epoch=0,
                             output_dir='plots'):
        logging.info(f"Visualisiere Embeddings => {method.upper()}, EPOCH={epoch}")
        embeddings_list = []
        combos = []
        for i, row in df.iterrows():
            pid = row['pid']
            study_yr = row['study_yr']
            combo = row['combination']

            emb = self.compute_patient_embedding(pid, study_yr)  # => (1,2048)
            emb_np = emb.squeeze(0).detach().cpu().numpy()
            embeddings_list.append(emb_np)
            combos.append(combo)

        embeddings_arr = np.array(embeddings_list)  # shape (N,2048)

        if method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            projector = TSNE(n_components=2, random_state=42)
        else:
            from sklearn.decomposition import PCA
            projector = PCA(n_components=2, random_state=42)

        coords_2d = projector.fit_transform(embeddings_arr)  # shape (N,2)

        plt.figure(figsize=(8,6))
        combos_unique = sorted(list(set(combos)))
        for c in combos_unique:
            idxs = [idx for idx, val in enumerate(combos) if val == c]
            plt.scatter(
                coords_2d[idxs, 0],
                coords_2d[idxs, 1],
                label=str(c),
                alpha=0.6
            )

        plt.title(f"{method.upper()} Embeddings - EPOCH={epoch}")
        plt.legend(loc='best')

        aggregator_name = getattr(self.mil_agg, '__class__').__name__
        timestamp = datetime.datetime.now().strftime("%m%d-%H%M")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = os.path.join(
            output_dir,
            f"Emb_{method}_{aggregator_name}_epoch{epoch:03d}_{timestamp}.png"
        )
        plt.savefig(filename, dpi=150)
        plt.close()
        logging.info(f"Saved embedding plot: {filename}")

    # --------------------------------------------------------------------------
    # plot_loss_curve => Speichert die Loss-Kurve als PNG
    # --------------------------------------------------------------------------
    def plot_loss_curve(self, output_dir='plots'):
        """
        Nutzt self.epoch_losses, um einen Loss-vs-Epoch Plot zu erstellen.
        Speichert loss_vs_epoch.png in output_dir.
        """
        if not self.epoch_losses:
            logging.info("Keine epoch_losses vorhanden, überspringe plot_loss_curve.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        epochs_range = list(range(1, len(self.epoch_losses)+1))
        plt.figure(figsize=(8,6))
        plt.plot(epochs_range, self.epoch_losses, marker='o', label="Train Loss", color='navy')
        plt.title("Progression of Training Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Triplet Loss")
        plt.legend()

        aggregator_name = getattr(self.mil_agg, '__class__').__name__
        timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
        filename = os.path.join(
            output_dir,
            f"loss_vs_epoch_{aggregator_name}_{timestamp}.png"
        )
        plt.savefig(filename, dpi=150)
        plt.close()
        logging.info(f"Saved loss curve: {filename}")

    # --------------------------------------------------------------------------
    # plot_metric_curves => Plottet Precision, Recall, mAP vs Epochen
    # --------------------------------------------------------------------------
    def plot_metric_curves(self, output_dir='plots'):
        """
        Zeichnet Precision@K, Recall@K, mAP über die Epochen als Liniendiagramme.
        Speichert sie gemeinsam in metrics_vs_epoch.png.
        """
        if len(self.metric_history["mAP"]) == 0:
            logging.info("Keine Metric-History vorhanden, überspringe plot_metric_curves.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        epochs_range = list(range(1, len(self.metric_history["mAP"])+1))

        plt.figure(figsize=(8,6))
        plt.plot(
            epochs_range,
            self.metric_history["precision"],
            marker='o',
            label="Precision@K",
            color='green'
        )
        plt.plot(
            epochs_range,
            self.metric_history["recall"],
            marker='s',
            label="Recall@K",
            color='orange'
        )
        plt.plot(
            epochs_range,
            self.metric_history["mAP"],
            marker='^',
            label="mAP",
            color='red'
        )

        plt.title("Progression of Precision@K, Recall@K, and mAP over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.ylim(0, 1.0)
        plt.legend(loc='best')

        aggregator_name = getattr(self.mil_agg, '__class__').__name__
        timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
        filename = os.path.join(
            output_dir,
            f"metrics_vs_epoch_{aggregator_name}_{timestamp}.png"
        )

        plt.savefig(filename, dpi=150)
        plt.close()
        logging.info(f"Saved metric curves: {filename}")

    # --------------------------------------------------------------------------
    # save_checkpoint / load_checkpoint
    # --------------------------------------------------------------------------
    def save_checkpoint(self, path):
        checkpoint = {
            'base_cnn': self.base_cnn.state_dict(),
            'mil_agg': self.mil_agg.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if self.scheduler is not None:
            checkpoint['scheduler'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logging.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.base_cnn.load_state_dict(checkpoint['base_cnn'])
        self.mil_agg.load_state_dict(checkpoint['mil_agg'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler is not None and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        logging.info(f"Checkpoint loaded from {path}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # # Beispiel:
    # data_csv = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\training\nlst_subset_v5_training.csv"
    # data_root = r"D:\thesis_robert\NLST_subset_v5_nifti_3mm_Voxel"
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
    #     weight_decay=1e-5
    # )

    # best_map, best_epoch = trainer.train_with_val(
    #     epochs=5,
    #     num_triplets=200,
    #     val_csv=r"C:\...\nlst_subset_v5_validation.csv",
    #     data_root_val=data_root,
    #     K=5,
    #     distance_metric='euclidean',
    #     visualize_every=2,
    #     visualize_method='pca',
    #     output_dir='plots_example'
    # )
    # logging.info(f"Done. best_map={best_map:.4f} at epoch={best_epoch}")
