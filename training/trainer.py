import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.utils.data import DataLoader

# 1) Projektpfad
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2) ResNet18-Feature Extractor
from model.base_cnn import BaseCNN

# 3) Gated-Attention MIL Aggregator (mit Dropout)
from model.mil_aggregator import AttentionMILAggregator

# 4) TripletSampler
from training.triplet_sampler import TripletSampler

# 5) SinglePatientDataset
from training.data_loader import SinglePatientDataset

import logging


class TripletTrainer(nn.Module):
    """
    Ein erweiterter Trainer, der nn.Module erbt, um state_dict() und load_state_dict() 
    nativ verwenden zu können. Enthält:
    - base_cnn (ResNet18)
    - mil_agg (Attention-MIL)
    - triplet_loss
    - optimizer
    - optional: scheduler
    - Methoden: train_loop, train_one_epoch, compute_patient_embedding, etc.
    - save_checkpoint / load_checkpoint
    """

    def __init__(self, 
                 df,
                 data_root,
                 device='cuda',
                 lr=1e-3, 
                 margin=1.0,
                 roi_size=(96,96,3),
                 overlap=(10,10,1),
                 pretrained=False,
                 attention_hidden_dim=128,
                 dropout=0.2,
                 weight_decay=1e-5):
        """
        Args:
            df: Pandas DataFrame mit Spalten [pid, study_yr, combination]
            data_root: Pfad zu den .nii.gz NIfTI-Daten
            device: 'cuda' oder 'cpu'
            lr: Lernrate
            margin: Triplet-Loss margin
            roi_size, overlap: Patch-Extraktions-Parameter
            pretrained: ob ResNet18 auf ImageNet-Weights basiert
            attention_hidden_dim: Größe der Hidden-Layer im Attention-MLP
            dropout: Dropout-Rate im Aggregator
            weight_decay: L2-Reg
        """
        super().__init__()  # wichtig, damit nn.Module initialisiert

        self.df = df
        self.data_root = data_root
        self.device = device

        self.roi_size = roi_size
        self.overlap = overlap

        # A) CNN-Backbone
        self.base_cnn = BaseCNN(model_name='resnet18', pretrained=pretrained).to(device)

        # B) Attention-MIL Aggregator
        # => 512 in base_cnn -> in_dim=512
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
        self.scheduler = None  # Kann später von außen gesetzt werden

        # store or print some logs
        logging.info(f"TripletTrainer init: lr={lr}, margin={margin}, dropout={dropout}, weight_decay={weight_decay}")

    # --------------------------------------------------------------------------
    # 1) Embedding-Berechnung ohne Grad (Inferenz)
    # --------------------------------------------------------------------------
    def compute_patient_embedding(self, pid, study_yr):
        """
        Lädt ALLE Patches für (pid, study_yr) => CNN => (N_patches,512) => AttentionMIL => (1,512).
        Keine Grad-Berechnung.
        """
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
                patch_t = patch_t.to(self.device)  # => (B,3,H,W)
                emb = self.base_cnn(patch_t)       # => (B,512)
                all_embs.append(emb)

        if len(all_embs) == 0:
            # Kein Patch => leeres embedding
            return torch.zeros((1,512), device=self.device)

        patch_embs = torch.cat(all_embs, dim=0)  # => (N_patches,512)
        patient_emb = self.mil_agg(patch_embs)   # => (1,512)
        return patient_emb

    # --------------------------------------------------------------------------
    # 2) Embedding-Berechnung mit Grad (Training)
    # --------------------------------------------------------------------------
    def _forward_patient(self, pid, study_yr):
        """
        Lädt alle Patches => CNN => Aggregation => (1,512) mit Grad.
        """
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
            # kein Patch => leeres embedding
            return torch.zeros((1,512), device=self.device, requires_grad=True)

        patch_embs = torch.cat(patch_embs, dim=0)
        patient_emb = self.mil_agg(patch_embs)  # => (1,512)
        return patient_emb

    # --------------------------------------------------------------------------
    # 3) train_one_epoch: Iteration über Sampler => TripletLoss
    # --------------------------------------------------------------------------
    def train_one_epoch(self, sampler):
        """
        sampler => Iteration über (Anchor,Pos,Neg). => 1 Triplet => 1 Optimizer-Schritt
        """
        total_loss = 0.0
        steps = 0

        for step, (anchor_info, pos_info, neg_info) in enumerate(sampler):
            anchor_pid, anchor_sy = anchor_info['pid'], anchor_info['study_yr']
            pos_pid, pos_sy = pos_info['pid'], pos_info['study_yr']
            neg_pid, neg_sy = neg_info['pid'], neg_info['study_yr']

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

    # --------------------------------------------------------------------------
    # 4) train_loop: Baut Sampler, läuft über num_epochs
    # --------------------------------------------------------------------------
    def train_loop(self, num_epochs=2, num_triplets=100):
        """
        Baut den TripletSampler und durchläuft num_epochs.
        Pro Epoche => train_one_epoch => sampler.reset_epoch
        Optional: scheduler.step() könnte man hier oder nach jeder Epoche aufrufen.
        """
        sampler = TripletSampler(
            df=self.df,
            num_triplets=num_triplets,
            shuffle=True,
            top_k_negatives=3
        )
        for epoch in range(1, num_epochs+1):
            logging.info(f"=== EPOCH {epoch}/{num_epochs} ===")
            self.train_one_epoch(sampler)

            # Wenn du einen Scheduler hast, kannst du hier self.scheduler.step()
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                logging.info(f"Nach Epoche {epoch}, LR={current_lr}")

            sampler.reset_epoch()

    # --------------------------------------------------------------------------
    # 5) Checkpoint speichern
    # --------------------------------------------------------------------------
    def save_checkpoint(self, path):
        """
        Speichert base_cnn, mil_agg, optimizer, optional scheduler
        als state_dict in path.
        """
        checkpoint = {
            'base_cnn': self.base_cnn.state_dict(),
            'mil_agg': self.mil_agg.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if self.scheduler is not None:
            checkpoint['scheduler'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logging.info(f"Checkpoint saved to {path}")

    # --------------------------------------------------------------------------
    # 6) Checkpoint laden
    # --------------------------------------------------------------------------
    def load_checkpoint(self, path):
        """
        Lädt base_cnn, mil_agg, optimizer, optional scheduler
        aus path.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.base_cnn.load_state_dict(checkpoint['base_cnn'])
        self.mil_agg.load_state_dict(checkpoint['mil_agg'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler is not None and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        logging.info(f"Checkpoint loaded from {path}")


# ------------------------------------------------------------------------------
# Falls du direkt trainer.py aufrufst (z.B. zu Debugging-Zwecken)
# ------------------------------------------------------------------------------
if __name__=="__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    data_csv = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\training\nlst_subset_v5_training.csv"
    data_root = r"D:\thesis_robert\NLST_subset_v5_nifti_3mm_Voxel"

    df = pd.read_csv(data_csv)

    trainer = TripletTrainer(
        df=df,
        data_root=data_root,
        device='cuda',
        lr=1e-4,
        margin=1.0,
        roi_size=(96,96,3),
        overlap=(10,10,1),
        pretrained=False,
        attention_hidden_dim=128,
        dropout=0.2,
        weight_decay=1e-5
    )

    # Debug-Training
    trainer.train_loop(num_epochs=2, num_triplets=200)
    # Example: Save
    trainer.save_checkpoint("debug_checkpoint.pt")
    # then load back
    trainer.load_checkpoint("debug_checkpoint.pt")
