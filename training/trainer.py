import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import datetime
import csv

from torch.utils.data import DataLoader

class TripletTrainerBase(nn.Module):
    """
    Abgespecktes Base Model:
     - ResNet18/50 (BaseCNN, param: model_name)
     - Optionale Freeze-Blocks
     - Attention-MIL (param: agg_hidden_dim, agg_dropout)
     - TripletMarginLoss
     - IR-Metriken (Precision@K, Recall@K, mAP)
     - CSV-Logging pro Epoche
    """

    def __init__(
        self,
        df,
        data_root,
        device='cuda',
        lr=1e-4,
        margin=1.0,
        model_name='resnet18',
        freeze_blocks=None,
        agg_hidden_dim=128,
        agg_dropout=0.2,
        roi_size=(96,96,3),
        overlap=(10,10,1)
    ):
        """
        Args:
          df: DataFrame mit Spalten: [pid, study_yr, combination]
          data_root: Ordner mit den .nii.gz-Dateien
          device: 'cuda' oder 'cpu'
          lr: Learning Rate
          margin: Triplet-Margin
          model_name: z.B. 'resnet18' oder 'resnet50'
          freeze_blocks: Liste oder None, z.B. [0,1] => ResNet-Layer1, Layer2
          agg_hidden_dim: Hidden-Dim im Attention-MLP
          agg_dropout: Dropout-Rate
          roi_size, overlap: Falls du Patches definieren möchtest
        """
        super().__init__()
        self.df = df
        self.data_root = data_root
        self.device = device
        self.lr = lr
        self.margin = margin
        self.model_name = model_name
        self.freeze_blocks = freeze_blocks
        self.agg_hidden_dim = agg_hidden_dim
        self.agg_dropout = agg_dropout
        self.roi_size = roi_size
        self.overlap = overlap

        # === 1) CNN-Backbone ===
        from model.base_cnn import BaseCNN
        self.base_cnn = BaseCNN(
            model_name=self.model_name,
            pretrained=True,
            freeze_blocks=self.freeze_blocks
        ).to(device)

        # === 2) Aggregator (Gated-Attention-MIL) ===
        from model.mil_aggregator import AttentionMILAggregator
        self.mil_agg = AttentionMILAggregator(
            in_dim=512,
            hidden_dim=self.agg_hidden_dim,
            dropout=self.agg_dropout
        ).to(device)

        # === 3) TripletLoss ===
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)

        # === 4) Optimizer ===
        params = list(self.base_cnn.parameters()) + list(self.mil_agg.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr)

        # === Tracking ===
        self.epoch_losses = []          # Gesamt-Loss pro Epoche
        self.epoch_triplet_losses = []  # Nur Triplet-Loss pro Epoche

        self.metric_history = {
            "precision": [],
            "recall": [],
            "mAP": []
        }

        # Best mAP
        self.best_val_map = 0.0
        self.best_val_epoch = -1

        logging.info("[TripletTrainerBase] Initialized")
        logging.info(f"lr={lr}, margin={margin}, model_name={model_name}, freeze_blocks={freeze_blocks}, "
                     f"agg_hidden_dim={agg_hidden_dim}, agg_dropout={agg_dropout}")

    def _forward_patient(self, pid, study_yr):
        """
        Lädt SinglePatientDataset => Patches => base_cnn => mil_agg => 1x512
        """
        from training.data_loader import SinglePatientDataset
        ds = SinglePatientDataset(
            data_root=self.data_root,
            pid=pid,
            study_yr=study_yr,
            roi_size=self.roi_size,
            overlap=self.overlap
        )
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        self.base_cnn.train()
        self.mil_agg.train()

        patch_embs = []
        for patch_t in loader:
            patch_t = patch_t.to(self.device)
            emb = self.base_cnn(patch_t)  # => (B,512)
            patch_embs.append(emb)

        if len(patch_embs)==0:
            # Edge case: kein Patch => Dummy
            dummy = torch.zeros((1,512), device=self.device, requires_grad=True)
            return dummy

        patch_embs = torch.cat(patch_embs, dim=0)   # (N,512)
        patient_emb = self.mil_agg(patch_embs)      # (1,512)
        return patient_emb

    def train_one_epoch(self, sampler):
        """
        Verwendet TripletSampler => (anchor_info, pos_info, neg_info)
        => Anchor, Positive, Negative => TripletLoss
        """
        total_loss = 0.0
        total_trip = 0.0
        steps = 0

        for step, (a_info, p_info, n_info) in enumerate(sampler):
            a_emb = self._forward_patient(a_info['pid'], a_info['study_yr'])
            p_emb = self._forward_patient(p_info['pid'], p_info['study_yr'])
            n_emb = self._forward_patient(n_info['pid'], n_info['study_yr'])

            trip_loss = self.triplet_loss_fn(a_emb, p_emb, n_emb)

            self.optimizer.zero_grad()
            trip_loss.backward()
            self.optimizer.step()

            total_loss += trip_loss.item()
            total_trip += trip_loss.item()
            steps += 1

            if step % 250 == 0:
                logging.info(f"[Step={step}] TripletLoss={trip_loss.item():.4f}")

        if steps>0:
            avg_loss = total_loss/steps
            avg_trip = total_trip/steps
        else:
            avg_loss, avg_trip = 0, 0

        self.epoch_losses.append(avg_loss)
        self.epoch_triplet_losses.append(avg_trip)

        logging.info(f"=> Epoche done, TripletLoss={avg_trip:.4f}")

    def train_loop(self, num_epochs=5, num_triplets=100):
        """
        Einfache Schleife:
         - TripletSampler
         - train_one_epoch
        """
        from training.triplet_sampler import TripletSampler
        sampler = TripletSampler(
            df=self.df,
            num_triplets=num_triplets,
            shuffle=True,
            top_k_negatives=3
        )

        for epoch in range(1, num_epochs+1):
            logging.info(f"=== EPOCH {epoch}/{num_epochs} ===")
            self.train_one_epoch(sampler)
            sampler.reset_epoch()

    def train_with_val(
        self,
        epochs,
        num_triplets,
        val_csv,
        data_root_val,
        K=3,
        distance_metric='euclidean',
        visualize_every=5,
        visualize_method='tsne',
        output_dir='plots',
        epoch_csv_path=None
    ):
        """
        Train + Evaluate IR-Metriken pro Epoche. Speichert pro Epoche in CSV.
        """
        if epoch_csv_path:
            if not os.path.exists(epoch_csv_path):
                with open(epoch_csv_path, mode='w', newline='') as f:
                    writer = csv.writer(f, delimiter=';')
                    # Wir hängen hier z.B. noch die Hyperparams in den Header an
                    header = [
                        "Epoch", "TotalLoss", "TripletLoss",
                        "Precision@K", "Recall@K", "mAP",
                        # Zusätzliche Felder, die z. B. im Header stehen
                        f"model_name={self.model_name}",
                        f"freeze_blocks={self.freeze_blocks}",
                        f"agg_hidden_dim={self.agg_hidden_dim}",
                        f"agg_dropout={self.agg_dropout}"
                    ]
                    writer.writerow(header)

        # Reset
        self.epoch_losses = []
        self.epoch_triplet_losses = []
        self.metric_history = {"precision": [], "recall": [], "mAP": []}
        self.best_val_map = 0.0
        self.best_val_epoch = -1

        df_val = pd.read_csv(val_csv)

        # Sampler
        from training.triplet_sampler import TripletSampler
        sampler = TripletSampler(
            df=self.df,
            num_triplets=num_triplets,
            shuffle=True,
            top_k_negatives=3
        )

        from evaluation.metrics import compute_embeddings, compute_precision_recall_map

        for epoch in range(1, epochs+1):
            logging.info(f"=== EPOCH {epoch}/{epochs} ===")
            # 1) train_one_epoch
            self.train_one_epoch(sampler)
            sampler.reset_epoch()

            # 2) Evaluate => IR-Metriken (Precision@K, Recall@K, mAP)
            emb_dict = compute_embeddings(
                trainer=self,
                df=df_val,
                data_root=data_root_val,
                device=self.device
            )
            val_metrics = compute_precision_recall_map(
                embeddings=emb_dict,
                K=K,
                distance_metric=distance_metric
            )
            precK = val_metrics["precision@K"]
            recK  = val_metrics["recall@K"]
            map_val = val_metrics["mAP"]

            logging.info(f"[Val-Epoch={epoch}] Precision@K={precK:.4f}, Recall@K={recK:.4f}, mAP={map_val:.4f}")
            self.metric_history["precision"].append(precK)
            self.metric_history["recall"].append(recK)
            self.metric_history["mAP"].append(map_val)

            if map_val>self.best_val_map:
                self.best_val_map = map_val
                self.best_val_epoch = epoch
                logging.info(f"=> New Best mAP={map_val:.4f} @ epoch={epoch}")

            # 3) Visualisierung (optional)
            if epoch % visualize_every == 0:
                self.visualize_embeddings(
                    df=df_val,
                    data_root=data_root_val,
                    method=visualize_method,
                    epoch=epoch,
                    output_dir=output_dir
                )

            # 4) CSV-Logging
            if epoch_csv_path:
                self._write_epoch_csv(epoch, epoch_csv_path)

        # Plot Loss + IR-Kurven
        self.plot_loss_components(output_dir=output_dir)
        self.plot_metric_curves(output_dir=output_dir)

        return (self.best_val_map, self.best_val_epoch)

    def _write_epoch_csv(self, epoch, csv_path):
        if len(self.epoch_losses) == 0:
            return
        total_loss = self.epoch_losses[-1]
        trip_loss  = self.epoch_triplet_losses[-1]

        precK = self.metric_history["precision"][-1]
        recK  = self.metric_history["recall"][-1]
        map_v = self.metric_history["mAP"][-1]

        row = [
            epoch,
            f"{total_loss:.4f}",
            f"{trip_loss:.4f}",
            f"{precK:.4f}",
            f"{recK:.4f}",
            f"{map_v:.4f}",
        ]

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(row)
        logging.info(f"=> Epoche {epoch} in CSV geloggt => {csv_path}")

    def compute_patient_embedding(self, pid, study_yr):
        """
        Für Evaluate => berechnet patient_emb (1,512) => IR-Metriken
        """
        from training.data_loader import SinglePatientDataset
        ds = SinglePatientDataset(
            data_root=self.data_root,
            pid=pid,
            study_yr=study_yr,
            roi_size=self.roi_size,
            overlap=self.overlap
        )
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        self.base_cnn.eval()
        self.mil_agg.eval()

        patch_embs = []
        with torch.no_grad():
            for patch_t in loader:
                patch_t = patch_t.to(self.device)
                emb = self.base_cnn(patch_t)
                patch_embs.append(emb)

        if len(patch_embs)==0:
            return torch.zeros((1,512), device=self.device)

        patch_embs = torch.cat(patch_embs, dim=0)  # (N,512)
        patient_emb = self.mil_agg(patch_embs)      # (1,512)
        return patient_emb

    def visualize_embeddings(self, df, data_root, method='tsne',
                             epoch=0, output_dir='plots'):
        import os
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        import numpy as np

        logging.info(f"Visualisiere Embeddings => {method.upper()}, EPOCH={epoch}")
        embeddings_list = []
        combos = []
        for i, row in df.iterrows():
            pid = row['pid']
            study_yr = row['study_yr']
            combo = row['combination']

            emb = self.compute_patient_embedding(pid, study_yr)
            emb_np = emb.squeeze(0).cpu().numpy()
            embeddings_list.append(emb_np)
            combos.append(combo)

        embeddings_arr = np.array(embeddings_list)
        if method.lower() == 'tsne':
            projector = TSNE(n_components=2, random_state=42)
        else:
            projector = PCA(n_components=2, random_state=42)

        coords_2d = projector.fit_transform(embeddings_arr)

        plt.figure(figsize=(8,6))
        unique_combos = sorted(list(set(combos)))
        for c in unique_combos:
            idxs = [ix for ix, val in enumerate(combos) if val==c]
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

    def plot_loss_components(self, output_dir='plots'):
        import matplotlib.pyplot as plt
        import os
        if not self.epoch_losses:
            logging.info("Keine Verlaufsdaten => skip plot_loss_components.")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        x_vals = range(1, len(self.epoch_losses)+1)

        plt.figure(figsize=(8,6))
        plt.plot(x_vals, self.epoch_losses,       'r-o', label="Total(Triplet) Loss")
        plt.title("Triplet Loss vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.legend()

        timestr = datetime.datetime.now().strftime("%m%d-%H%M")
        outname = os.path.join(output_dir, f"triplet_loss_vs_epoch_{timestr}.png")
        plt.savefig(outname, dpi=150)
        plt.close()
        logging.info(f"Saved triplet loss plot: {outname}")

    def plot_metric_curves(self, output_dir='plots'):
        import matplotlib.pyplot as plt
        import os
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
        plt.title("Precision@K, Recall@K, mAP vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()

        timestr = datetime.datetime.now().strftime("%m%d-%H%M")
        outname = os.path.join(output_dir, f"ir_metrics_vs_epoch_{timestr}.png")
        plt.savefig(outname, dpi=150)
        plt.close()
        logging.info(f"Saved IR metrics plot: {outname}")

    def save_checkpoint(self, path):
        ckpt = {
            'base_cnn': self.base_cnn.state_dict(),
            'mil_agg': self.mil_agg.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(ckpt, path)
        logging.info(f"Checkpoint saved => {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.base_cnn.load_state_dict(ckpt['base_cnn'])
        self.mil_agg.load_state_dict(ckpt['mil_agg'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        logging.info(f"Checkpoint loaded from {path}")
