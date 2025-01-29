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

# 1) CNN-Backbone
from model.base_cnn import BaseCNN
# 2) Aggregatoren
from model.mil_aggregator import AttentionMILAggregator
from model.max_pooling import MaxPoolingAggregator
from model.mean_pooling import MeanPoolingAggregator
# 3) TripletSampler
from training.triplet_sampler import TripletSampler
# 4) SinglePatientDataset
from training.data_loader import SinglePatientDataset
# 5) Evaluate-Funktionen (IR-Metriken)
from evaluation.metrics import evaluate_model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def parse_combo_str_to_vec(combo_str):
    """
    '0-1-1' => [0,1,1]
    """
    return [int(x) for x in combo_str.split('-')]

class TripletTrainer(nn.Module):
    def __init__(
        self,
        aggregator_name,
        df,
        data_root,
        device='cuda',
        lr=1e-3,
        margin=1.0,
        roi_size=(96, 96, 3),
        overlap=(10,10,1),
        pretrained=False,
        attention_hidden_dim=128,
        dropout=0.2,
        weight_decay=1e-5,
        freeze_blocks=None,
        # SinglePatientDataset
        skip_slices=True,
        skip_factor=2,
        filter_empty_patches=False,
        min_nonzero_fraction=0.01,
        filter_uniform_patches=True,
        min_std_threshold=0.01,
        do_patch_minmax=False,
        # Hybrid-Loss
        lambda_bce=1.0,
        # NEU: manueller Parameter für Augmentation im Training
        do_augmentation_train=True
    ):
        super().__init__()
        self.df = df
        self.data_root = data_root
        self.device = device
        self.roi_size = roi_size
        self.overlap = overlap

        self.skip_slices = skip_slices
        self.skip_factor = skip_factor
        self.filter_empty_patches = filter_empty_patches
        self.min_nonzero_fraction = min_nonzero_fraction
        self.filter_uniform_patches = filter_uniform_patches
        self.min_std_threshold = min_std_threshold
        self.do_patch_minmax = do_patch_minmax

        self.lambda_bce = lambda_bce

        # NEU: manuell gesteuert
        self.do_augmentation_train = do_augmentation_train


        # (A) CNN-Backbone
        self.base_cnn = BaseCNN(
            model_name='resnet18',
            pretrained=pretrained,
            freeze_blocks=freeze_blocks
        ).to(device)

        # (B) Aggregator
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

        # (D) Multi-Label-Kopf + BCE-Loss
        self.classifier = nn.Linear(512, 3).to(device)
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

        # (E) Optimizer
        params = list(self.base_cnn.parameters()) + \
                 list(self.mil_agg.parameters()) + \
                 list(self.classifier.parameters())

        self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

        # (F) Scheduler (optional)
        self.scheduler = None

        # Tracking
        self.best_val_map = 0.0
        self.best_val_epoch = -1

        # Verlaufslisten
        self.epoch_losses = []          # Gesamtloss (Triplet + BCE)
        self.epoch_triplet_losses = []  # Nur Triplet
        self.epoch_bce_losses = []      # Nur BCE

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
            f"filter_uniform_patches={filter_uniform_patches}, min_std_threshold={min_std_threshold}, "
            f"do_patch_minmax={do_patch_minmax}, lambda_bce={lambda_bce}, "
            f"do_augmentation_train={do_augmentation_train}"
        )

    # ---------------------------
    # _forward_patient => (emb, logits)
    # ---------------------------
    def _forward_patient(self, pid, study_yr):
        """
        => Nur beim Training. do_augmentation_train steuert Aug.
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
            filter_uniform_patches=self.filter_uniform_patches,
            min_std_threshold=self.min_std_threshold,
            do_patch_minmax=self.do_patch_minmax,
            do_augmentation=self.do_augmentation_train  # <-- hier
        )

        loader = DataLoader(ds, batch_size=32, shuffle=False)

        self.base_cnn.train()
        self.mil_agg.train()
        self.classifier.train()

        patch_embs = []
        for patch_t in loader:
            patch_t = patch_t.to(self.device)
            emb = self.base_cnn(patch_t)
            patch_embs.append(emb)

        if len(patch_embs)==0:
            dummy = torch.zeros((1,512), device=self.device, requires_grad=True)
            dummy_logits = self.classifier(dummy)
            return dummy, dummy_logits

        patch_embs = torch.cat(patch_embs, dim=0)
        patient_emb = self.mil_agg(patch_embs)
        logits = self.classifier(patient_emb)
        return patient_emb, logits

    def compute_patient_embedding(self, pid, study_yr):
        """
        => Validation / Test => do_augmentation=False
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
            filter_uniform_patches=self.filter_uniform_patches,
            min_std_threshold=self.min_std_threshold,
            do_patch_minmax=self.do_patch_minmax,
            do_augmentation=False  # Kein Aug. im Val/Test
        )
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        self.base_cnn.eval()
        self.mil_agg.eval()
        self.classifier.eval()

        all_embs = []
        with torch.no_grad():
            for patch_t in loader:
                patch_t = patch_t.to(self.device)
                emb = self.base_cnn(patch_t)
                all_embs.append(emb)

        if len(all_embs)==0:
            return torch.zeros((1,512), device=self.device)

        patch_embs = torch.cat(all_embs, dim=0)
        patient_emb = self.mil_agg(patch_embs)
        return patient_emb

    # ---------------------------
    # train_one_epoch
    # ---------------------------
    def train_one_epoch(self, sampler):
        total_loss = 0.0
        total_trip = 0.0
        total_bce  = 0.0
        steps = 0

        for step, (anchor_info, pos_info, neg_info) in enumerate(sampler):
            # Infos
            a_pid, a_sy = anchor_info['pid'], anchor_info['study_yr']
            p_pid, p_sy = pos_info['pid'], pos_info['study_yr']
            n_pid, n_sy = neg_info['pid'], neg_info['study_yr']

            a_label = anchor_info['multi_label']  # z.B. [0,1,1]
            p_label = pos_info['multi_label']
            n_label = neg_info['multi_label']

            # Forward
            a_emb, a_logits = self._forward_patient(a_pid, a_sy)
            p_emb, p_logits = self._forward_patient(p_pid, p_sy)
            n_emb, n_logits = self._forward_patient(n_pid, n_sy)

            # TripletLoss
            trip_loss = self.triplet_loss_fn(a_emb, p_emb, n_emb)

            # BCE-Loss
            a_label_t = torch.tensor(a_label, dtype=torch.float32, device=self.device).unsqueeze(0)
            p_label_t = torch.tensor(p_label, dtype=torch.float32, device=self.device).unsqueeze(0)
            n_label_t = torch.tensor(n_label, dtype=torch.float32, device=self.device).unsqueeze(0)

            bce_a = self.bce_loss_fn(a_logits, a_label_t)
            bce_p = self.bce_loss_fn(p_logits, p_label_t)
            bce_n = self.bce_loss_fn(n_logits, n_label_t)
            bce_loss = (bce_a + bce_p + bce_n)/3.0

            loss = trip_loss + self.lambda_bce*bce_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_trip += trip_loss.item()
            total_bce  += bce_loss.item()
            steps += 1

            if step % 250 == 0:
                logging.info(f"[Step {step}] TotalLoss={loss.item():.4f}  Triplet={trip_loss.item():.4f}  BCE={bce_loss.item():.4f}")

        if steps>0:
            avg_loss = total_loss/steps
            avg_trip = total_trip/steps
            avg_bce  = total_bce/steps
        else:
            avg_loss=0; avg_trip=0; avg_bce=0

        logging.info(f"Epoch Loss = {avg_loss:.4f} (Trip={avg_trip:.4f}, BCE={avg_bce:.4f})")

        # Speichern
        self.epoch_losses.append(avg_loss)
        self.epoch_triplet_losses.append(avg_trip)
        self.epoch_bce_losses.append(avg_bce)

    # ---------------------------
    # train_loop
    # ---------------------------
    def train_loop(self, num_epochs=2, num_triplets=100):
        sampler = TripletSampler(
            df=self.df,
            num_triplets=num_triplets,
            shuffle=True,
            top_k_negatives=3
        )
        for epoch in range(1, num_epochs+1):
            self.train_one_epoch(sampler)

            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                logging.info(f"Nach Epoche {epoch}, LR={current_lr}")

            sampler.reset_epoch()

    # ---------------------------
    # train_with_val => + evaluate
    # ---------------------------
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

        # Leeren
        self.epoch_losses = []
        self.epoch_triplet_losses = []
        self.epoch_bce_losses = []
        self.metric_history = {"precision": [], "recall": [], "mAP": []}

        # OPTIONAL: Multi-Label-Tracking über Epochen
        self.multilabel_history = {
            "fibrose_f1": [],
            "emphysem_f1": [],
            "nodule_f1": [],
            "macro_f1": []
        }

        df_val = pd.read_csv(val_csv)

        for epoch in range(1, epochs+1):
            logging.info(f"=== EPOCH {epoch}/{epochs} ===")
            # 1 Epoche train
            self.train_loop(num_epochs=1, num_triplets=num_triplets)

            # 2) Evaluate => IR-Metriken
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

            # Track best IR-mAP
            if map_val > self.best_val_map:
                self.best_val_map = map_val
                self.best_val_epoch = epoch
                logging.info(f"=> New Best mAP={map_val:.4f} @ epoch={epoch}")

            # 3) Evaluate => Multi-Label-Classification
            #    z. B. fibrose/emphysem/nodule
            multilabel_results = self.evaluate_multilabel_classification(df_val, data_root_val, threshold=0.5)
            logging.info(f"Multi-Label => {multilabel_results}")
            # => z. B. {'fibrose_precision': 0.87, 'fibrose_recall': 0.64, 'fibrose_f1': 0.74, ... }

            # OPTIONAL: Speichere F1 in deinen Verlaufsdict
            self.multilabel_history["fibrose_f1"].append(multilabel_results["fibrose_f1"])
            self.multilabel_history["emphysem_f1"].append(multilabel_results["emphysem_f1"])
            self.multilabel_history["nodule_f1"].append(multilabel_results["nodule_f1"])
            self.multilabel_history["macro_f1"].append(multilabel_results["macro_f1"])

            # 4) Visualisierung
            if epoch % visualize_every == 0:
                self.visualize_embeddings(
                    df=df_val,
                    data_root=data_root_val,
                    method=visualize_method,
                    epoch=epoch,
                    output_dir=output_dir
                )

        # Plot Loss-Kurven
        self.plot_loss_components(output_dir=output_dir)

        # Plot IR-Metriken
        self.plot_metric_curves(output_dir=output_dir)

        # OPTIONAL: Du könntest hier jetzt auch noch deine multi-label-F1 vs. Epoche plotten
        self.plot_multilabel_f1_curves(output_dir=output_dir)

        return (self.best_val_map, self.best_val_epoch)


    # ---------------------------
    # Multi-Label Evaluate
    # ---------------------------
    def evaluate_multilabel_classification(self, df, data_root, threshold=0.5):
        """
        1) Für jeden Patienten => logits => sigmoid => predicted 0/1
        2) Mit df['combination'] vergleichen => [0,1,1]
        3) Precision, Recall, F1 je Merkmal => + Macro-F1
        """
        self.base_cnn.eval()
        self.mil_agg.eval()
        self.classifier.eval()

        # Stats pro Merkmal
        # fibrose => idx=0
        # emphysem => idx=1
        # nodule => idx=2
        # Zähle TP,FP,FN pro Merkmal
        TP = [0,0,0]
        FP = [0,0,0]
        FN = [0,0,0]

        for idx, row in df.iterrows():
            pid = row['pid']
            study_yr = row['study_yr']
            combo_str = row['combination']
            gt = parse_combo_str_to_vec(combo_str)  # [0,1,1]

            # Vorhersage: => embedding => logits => sigmoid
            with torch.no_grad():
                emb = self.compute_patient_embedding(pid, study_yr)
                # => shape (1,512)
                logits = self.classifier(emb)
                # => shape (1,3)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                # => shape (3,)

            pred = [1 if p>=threshold else 0 for p in probs]

            # Zähle TP,FP,FN pro Index
            for i in range(3):
                if gt[i]==1 and pred[i]==1:
                    TP[i]+=1
                elif gt[i]==0 and pred[i]==1:
                    FP[i]+=1
                elif gt[i]==1 and pred[i]==0:
                    FN[i]+=1

        # Compute P,R,F1 pro Merkmal
        results = {}
        sum_f1 = 0.0
        for i, name in enumerate(["fibrose","emphysem","nodule"]):
            precision_i = TP[i] / (TP[i]+FP[i]) if (TP[i]+FP[i])>0 else 0
            recall_i    = TP[i] / (TP[i]+FN[i]) if (TP[i]+FN[i])>0 else 0
            f1_i = 0.0
            if precision_i+recall_i>0:
                f1_i = 2.0*precision_i*recall_i/(precision_i+recall_i)

            results[f"{name}_precision"] = precision_i
            results[f"{name}_recall"]    = recall_i
            results[f"{name}_f1"]        = f1_i
            sum_f1 += f1_i

        # Macro-F1 => average f1 across 3
        macro_f1 = sum_f1/3.0
        results["macro_f1"] = macro_f1
        return results

    # ---------------------------
    # VIS
    # ---------------------------
    def visualize_embeddings(self, df, data_root,
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

            emb = self.compute_patient_embedding(pid, study_yr)
            emb_np = emb.squeeze(0).detach().cpu().numpy()
            embeddings_list.append(emb_np)
            combos.append(combo)

        embeddings_arr = np.array(embeddings_list)
        if method.lower()=='tsne':
            from sklearn.manifold import TSNE
            projector = TSNE(n_components=2, random_state=42)
        else:
            from sklearn.decomposition import PCA
            projector = PCA(n_components=2, random_state=42)

        coords_2d = projector.fit_transform(embeddings_arr)

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

    # ---------------------------
    # plot_loss_components
    # ---------------------------
    def plot_loss_components(self, output_dir='plots'):
        """
        Zeichnet TripletLoss, BCE-Loss und Gesamt-Loss über die Epochen.
        """
        if not self.epoch_losses:
            logging.info("Keine Verlaufsdaten => skip plot_loss_components.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        epochs_range = range(1, len(self.epoch_losses)+1)

        plt.figure(figsize=(8,6))
        plt.plot(epochs_range, self.epoch_losses,    'r-o', label="Total Loss")
        plt.plot(epochs_range, self.epoch_triplet_losses, 'b-^', label="Triplet Loss")
        plt.plot(epochs_range, self.epoch_bce_losses,     'g-s', label="BCE Loss")

        plt.title("Loss Components vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.legend()

        aggregator_name = getattr(self.mil_agg, '__class__').__name__
        timestr = datetime.datetime.now().strftime("%m%d-%H%M")
        outname = os.path.join(output_dir, f"loss_components_{aggregator_name}_{timestr}.png")

        plt.savefig(outname, dpi=150)
        plt.close()
        logging.info(f"Saved loss components plot: {outname}")

    # ---------------------------
    # plot_metric_curves
    # ---------------------------
    def plot_metric_curves(self, output_dir='plots'):
        if len(self.metric_history["mAP"])==0:
            logging.info("Keine IR-Metriken => skip plot_metric_curves.")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        x_vals = list(range(1, len(self.metric_history["mAP"])+1))

        plt.figure(figsize=(8,6))
        plt.plot(x_vals, self.metric_history["precision"], 'g-o', label='Precision@K')
        plt.plot(x_vals, self.metric_history["recall"],    'm-s', label='Recall@K')
        plt.plot(x_vals, self.metric_history["mAP"],       'r-^', label='mAP')
        plt.ylim(0,1)
        plt.title("Precision@K, Recall@K, mAP vs. Epoche")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()

        aggregator_name = getattr(self.mil_agg, '__class__').__name__
        timestr = datetime.datetime.now().strftime("%m%d-%H%M")
        outname = os.path.join(output_dir, f"metrics_vs_epoch_{aggregator_name}_{timestr}.png")
        plt.savefig(outname, dpi=150)
        plt.close()
        logging.info(f"Saved metric curves: {outname}")

    def plot_multilabel_f1_curves(self, output_dir='plots'):
        if not hasattr(self, 'multilabel_history'):
            logging.info("No multilabel_history => skip plot_multilabel_f1_curves.")
            return
        if len(self.multilabel_history["macro_f1"]) == 0:
            logging.info("No data in multilabel_history => skip plot_multilabel_f1_curves.")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        x_vals = range(1, len(self.multilabel_history["macro_f1"])+1)

        fib_f1   = self.multilabel_history["fibrose_f1"]
        emph_f1  = self.multilabel_history["emphysem_f1"]
        nod_f1   = self.multilabel_history["nodule_f1"]
        mac_f1   = self.multilabel_history["macro_f1"]

        plt.figure(figsize=(8,6))
        plt.plot(x_vals, fib_f1,   'r-o', label='Fibrose F1')
        plt.plot(x_vals, emph_f1,  'g-s', label='Emphysem F1')
        plt.plot(x_vals, nod_f1,   'b-^', label='Nodule F1')
        plt.plot(x_vals, mac_f1,   'k--', label='Macro-F1', linewidth=2.0)

        plt.title("Multi-Label F1 vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.ylim(0,1)
        plt.legend(loc='best')

        aggregator_name = getattr(self.mil_agg, '__class__').__name__
        timestr = datetime.datetime.now().strftime("%m%d-%H%M")
        outname = os.path.join(output_dir, f"multilabel_f1_vs_epoch_{aggregator_name}_{timestr}.png")

        plt.savefig(outname, dpi=150)
        plt.close()
        logging.info(f"Saved multi-label F1 curve: {outname}")


    # ---------------------------
    # Checkpoint
    # ---------------------------
    def save_checkpoint(self, path):
        ckpt = {
            'base_cnn': self.base_cnn.state_dict(),
            'mil_agg': self.mil_agg.state_dict(),
            'classifier': self.classifier.state_dict(),
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
        self.classifier.load_state_dict(ckpt['classifier'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

        if self.scheduler is not None and 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])

        logging.info(f"Checkpoint loaded from {path}")
