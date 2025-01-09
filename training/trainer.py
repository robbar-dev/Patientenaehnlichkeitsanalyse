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

# 3) Importiere Deinen AttentionMILAggregator
from model.mil_aggregator import AttentionMILAggregator

# 4) TripletSampler
from training.triplet_sampler import TripletSampler

# 5) SinglePatientDataset
from training.data_loader import SinglePatientDataset

class TripletTrainer:
    def __init__(self, df, data_root, device='cuda', lr=1e-3, margin=1.0,
                 roi_size=(96,96,3), overlap=(10,10,1), pretrained=False,
                 attention_hidden_dim=128):
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
        """
        self.df = df
        self.data_root = data_root
        self.device = device
        self.roi_size = roi_size
        self.overlap = overlap

        # (A) CNN-Backbone
        self.base_cnn = BaseCNN(model_name='resnet18', pretrained=pretrained).to(device)

        # (B) Attention-MIL Aggregator
        #    Falls Du die Dimension 512 in base_cnn hast, setze in_dim=512
        self.mil_agg = AttentionMILAggregator(in_dim=512, hidden_dim=128, dropout=0.2).to(device)

        # (C) TripletLoss
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

        # (D) Optimizer (inkl. Aggregator-Params)
        self.optimizer = optim.Adam(
            list(self.base_cnn.parameters()) + list(self.mil_agg.parameters()),
            lr=lr
        )

    def compute_patient_embedding(self, pid, study_yr):
        """
        Lädt ALLE Patches für (pid, study_yr) => CNN => (N_patches,512) => AttentionMIL => (1,512).
        (keine Grad, reine Inferenz)
        """
        ds = SinglePatientDataset(
            data_root=self.data_root,
            pid=pid,
            study_yr=study_yr,
            roi_size=self.roi_size,
            overlap=self.overlap
        )
        loader = DataLoader(ds, batch_size=16, shuffle=False)

        all_embs = []
        self.base_cnn.eval()
        self.mil_agg.eval()

        with torch.no_grad():
            for patch_t in loader:
                patch_t = patch_t.to(self.device)  # => (B,3,H,W)
                emb = self.base_cnn(patch_t)       # => (B,512)
                all_embs.append(emb)

        if len(all_embs) == 0:
            # kein Patch => leeres embedding
            return torch.zeros((1,512), device=self.device)

        patch_embs = torch.cat(all_embs, dim=0)  # => (N_patches,512)
        patient_emb = self.mil_agg(patch_embs)   # => (1,512)
        return patient_emb

    def _forward_patient(self, pid, study_yr):
        """
        Variante mit Grad => wir trainieren BN/Dropout => (train-mode).
        """
        ds = SinglePatientDataset(
            data_root=self.data_root,
            pid=pid,
            study_yr=study_yr,
            roi_size=self.roi_size,
            overlap=self.overlap
        )
        loader = DataLoader(ds, batch_size=16, shuffle=False)

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

    def train_one_epoch(self, sampler):
        """
        sampler => Iteration über (A_info, P_info, N_info).
        => pro Iteration 1 Triplet => 1 Optimizer-Schritt
        """
        total_loss = 0.0
        steps = 0

        for step, (anchor_info, pos_info, neg_info) in enumerate(sampler):
            anchor_pid, anchor_sy = anchor_info['pid'], anchor_info['study_yr']
            pos_pid, pos_sy = pos_info['pid'], pos_info['study_yr']
            neg_pid, neg_sy = neg_info['pid'], neg_info['study_yr']

            # => Embeddings mit Grad
            anchor_emb = self._forward_patient(anchor_pid, anchor_sy)
            pos_emb    = self._forward_patient(pos_pid, pos_sy)
            neg_emb    = self._forward_patient(neg_pid, neg_sy)

            loss = self.triplet_loss_fn(anchor_emb, pos_emb, neg_emb)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

            if step % 50 == 0:
                print(f"[Step {step}] Triplet Loss = {loss.item():.4f}")

        avg_loss = total_loss / steps if steps>0 else 0.0
        print(f"Epoch Loss = {avg_loss:.4f}")

    def train_loop(self, num_epochs=2, num_triplets=100):
        """
        Baut den TripletSampler und durchläuft num_epochs
        """
        sampler = TripletSampler(
            df=self.df,
            num_triplets=num_triplets,
            shuffle=True,
            top_k_negatives=3
        )
        for epoch in range(num_epochs):
            print(f"=== EPOCH {epoch+1}/{num_epochs} ===")
            self.train_one_epoch(sampler)
            # optional: sampler.reset_epoch() => in der nächsten Epoche erneut selbst. Triplets


if __name__=="__main__":
    data_csv = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\nlst_subset_v5.csv"
    data_root = r"D:\thesis_robert\NLST_subset_v5_nifti_3mm_Voxel"

    df = pd.read_csv(data_csv)

    # Starte das Training
    trainer = TripletTrainer(
        df=df,
        data_root=data_root,
        device='cuda',
        lr=1e-4,
        margin=1.0,
        roi_size=(96,96,3),
        overlap=(10,10,1),
        pretrained=False,
        attention_hidden_dim=128
    )

    trainer.train_loop(num_epochs=10, num_triplets=1000)
