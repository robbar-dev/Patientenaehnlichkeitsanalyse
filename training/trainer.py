import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.base_cnn import BaseCNN
from model.mil_aggregator import AttentionMILAggregator
from training.triplet_sampler import TripletSampler
from training.data_loader import SinglePatientDataset

# Neu für Visualisierung
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class TripletTrainer(nn.Module):
    """
    Erweiterter Trainer mit:
      - train_with_val() + Visualisierung der Embeddings alle 5 Epochen
    """

    def __init__(self,
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
                 weight_decay=1e-5):
        super().__init__()
        self.df = df
        self.data_root = data_root
        self.device = device
        self.roi_size = roi_size
        self.overlap = overlap

        # A) CNN-Backbone
        self.base_cnn = BaseCNN(model_name='resnet18', pretrained=pretrained).to(device)

        # B) Attention-MIL Aggregator
        self.mil_agg = AttentionMILAggregator(
            in_dim=512,
            hidden_dim=attention_hidden_dim,
            dropout=dropout
        ).to(device)

        # C) TripletLoss
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

        # D) Optimizer
        self.optimizer = optim.Adam(
            list(self.base_cnn.parameters()) + list(self.mil_agg.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        # E) Optional: Scheduler
        self.scheduler = None

        # Attribute für Best-Val
        self.best_val_map = 0.0
        self.best_val_epoch = -1

        logging.info(f"TripletTrainer init: lr={lr}, margin={margin}, dropout={dropout}, weight_decay={weight_decay}")

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
                emb = self.base_cnn(patch_t)  # => (B,512)
                all_embs.append(emb)

        if len(all_embs) == 0:
            return torch.zeros((1,512), device=self.device)

        patch_embs = torch.cat(all_embs, dim=0)  # => (N,512)
        patient_emb = self.mil_agg(patch_embs)   # => (1,512)
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
            emb = self.base_cnn(patch_t)  # => (B,512)
            patch_embs.append(emb)

        if len(patch_embs) == 0:
            return torch.zeros((1,512), device=self.device, requires_grad=True)

        patch_embs = torch.cat(patch_embs, dim=0)
        patient_emb = self.mil_agg(patch_embs)  # => (1,512)
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

        avg_loss = total_loss / steps if steps>0 else 0.0
        logging.info(f"Epoch Loss = {avg_loss:.4f}")

    # --------------------------------------
    # train_loop => Schleife
    # --------------------------------------
    def train_loop(self, num_epochs=2, num_triplets=100):
        sampler = TripletSampler(
            df=self.df,
            num_triplets=num_triplets,
            shuffle=True,
            top_k_negatives=3
        )
        for epoch in range(1, num_epochs+1):
            logging.info(f"=== EPOCH {epoch}/{num_epochs} ===")
            self.train_one_epoch(sampler)
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                logging.info(f"Nach Epoche {epoch}, LR={current_lr}")
            sampler.reset_epoch()

    # --------------------------------------------------------------------------
    # (NEU) train_with_val => Train + Val + Visualisierung
    # --------------------------------------------------------------------------
    def train_with_val(self,
                       epochs,
                       num_triplets,
                       val_csv,
                       data_root_val,
                       K=5,
                       distance_metric='euclidean',
                       visualize_every=5,
                       visualize_method='tsne',
                       output_dir='plots'):
        """
        Führt ein Training + Validierung durch.
        - Alle 'epochs'
        - pro Epoche => train_loop(num_epochs=1)
        - anschließend Evaluate + Track best_val_map
        - optional: Visualisierung Embeddings alle 'visualize_every' Epochen

        Returns:
            (best_val_map, best_val_epoch)
        """
        from evaluation.metrics import evaluate_model

        self.best_val_map = 0.0
        self.best_val_epoch = -1

        # Damit wir das val_df zur Visualisierung haben:
        df_val = pd.read_csv(val_csv)

        for epoch in range(1, epochs+1):
            logging.info(f"\n=== EPOCH {epoch}/{epochs} (Train) ===")
            # 1 Epoche train
            self.train_loop(num_epochs=1, num_triplets=num_triplets)

            # Evaluate
            logging.info(f"=== EPOCH {epoch}/{epochs} (Validation) ===")
            val_metrics = evaluate_model(
                trainer=self,
                data_csv=val_csv,
                data_root=data_root_val,
                K=K,
                distance_metric=distance_metric,
                device=self.device
            )
            current_map = val_metrics['mAP']
            logging.info(f"Val-Epoch={epoch}: mAP={current_map:.4f}")

            # Track Best
            if current_map > self.best_val_map:
                self.best_val_map = current_map
                self.best_val_epoch = epoch
                logging.info(f"=> Neuer Best mAP={self.best_val_map:.4f} in Epoche {self.best_val_epoch}")

            # Visualisierung => alle x Epochen
            if epoch % visualize_every == 0:
                self.visualize_embeddings(
                    df=df_val,
                    data_root=data_root_val,
                    method=visualize_method,
                    epoch=epoch,
                    output_dir=output_dir
                )

        return (self.best_val_map, self.best_val_epoch)

    # --------------------------------------------------------------------------
    # Visualisierung der Embeddings => t-SNE oder PCA
    # --------------------------------------------------------------------------
    def visualize_embeddings(self,
                             df,
                             data_root,
                             method='tsne',
                             epoch=0,
                             output_dir='plots'):
        """
        Berechnet Embeddings für alle PIDs in df, führt t-SNE oder PCA auf 2D durch,
        zeichnet Scatter-Plot mit Färbung nach 'combination', speichert PNG.

        df: DataFrame [pid, study_yr, combination]
        method: 'tsne' oder 'pca'
        epoch: Epochenzahl (für Dateinamen)
        output_dir: Wohin PNG gespeichert wird
        """
        logging.info(f"Visualisiere Embeddings => {method.upper()}, EPOCH={epoch}")
        # 1) Alle Embeddings + combos sammeln
        embeddings_list = []
        combos = []
        for i, row in df.iterrows():
            pid = row['pid']
            study_yr = row['study_yr']
            combo = row['combination']

            emb = self.compute_patient_embedding(pid, study_yr)  # => (1,512)
            emb_np = emb.squeeze(0).cpu().numpy()  # => (512,)
            embeddings_list.append(emb_np)
            combos.append(combo)

        embeddings_arr = np.array(embeddings_list)  # shape (N,512)

        # 2) Reduktion => PCA oder TSNE
        if method.lower() == 'tsne':
            projector = TSNE(n_components=2, random_state=42)
        else:
            projector = PCA(n_components=2, random_state=42)

        coords_2d = projector.fit_transform(embeddings_arr)  # shape (N,2)

        # 3) Plot
        plt.figure(figsize=(8,6))
        combos_unique = sorted(list(set(combos)))
        # Farbkodierung pro combo
        for c in combos_unique:
            idxs = [idx for idx, val in enumerate(combos) if val == c]
            plt.scatter(coords_2d[idxs, 0],
                        coords_2d[idxs, 1],
                        label=str(c),
                        alpha=0.6)

        plt.title(f"{method.upper()} Embeddings - EPOCH={epoch}")
        plt.legend(loc='best')
        # 4) Speichern
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = os.path.join(output_dir, f"Emb_{method}_{epoch:03d}.png")
        plt.savefig(filename, dpi=150)
        plt.close()
        logging.info(f"Saved embedding plot: {filename}")

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

    # # Kurzer Debug-Test
    # data_csv = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\training\nlst_subset_v5_training.csv"
    # data_root = r"D:\thesis_robert\NLST_subset_v5_nifti_3mm_Voxel"
    # df = pd.read_csv(data_csv)

    # trainer = TripletTrainer(
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

    # # Beispiel: train_with_val => 2 Epochen, Visualisierung alle 1 Epoche
    # best_map, best_epoch = trainer.train_with_val(
    #     epochs=2,
    #     num_triplets=200,
    #     val_csv=r"C:\...\nlst_subset_v5_validation.csv",
    #     data_root_val=data_root,
    #     K=5,
    #     distance_metric='euclidean',
    #     visualize_every=1,         # alle 1 Epochen => debug
    #     visualize_method='pca',    # oder 'tsne'
    #     output_dir='plots_debug'
    # )

    # logging.info(f"FERTIG - best_map={best_map:.4f} epoch={best_epoch}")
