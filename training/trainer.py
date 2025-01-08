import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.utils.data import DataLoader

# Füge Dein Projektverzeichnis hinzu (falls nötig)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
# 1) ResNet18-Feature Extractor
from model.base_cnn import BaseCNN

# 2) MIL Aggregator: simplest (Mean)
class MILAggregator(nn.Module):
    def forward(self, patch_embs):
        # patch_embs: (N_patches, 512)
        # => (1, 512)
        return patch_embs.mean(dim=0, keepdim=True)

# 3) TripletSampler
from training.triplet_sampler import TripletSampler

# 4) SinglePatientDataset
from training.data_loader import SinglePatientDataset


class TripletTrainer:
    def __init__(self, df, data_root, device='cuda', lr=1e-3, margin=1.0,
                 roi_size=(96,96,3), overlap=(10,10,1), pretrained=False):
        """
        Args:
            df: Pandas DataFrame mit Spalten [pid, study_yr, combination]
            data_root: Pfad zu den .nii.gz NIfTI-Daten
            device: 'cuda' oder 'cpu'
            lr: Lernrate
            margin: Triplet-Margin
            roi_size, overlap: Patch-Extraktions-Parameter
            pretrained: ob ResNet18 auf ImageNet-Weights basiert
        """
        self.df = df
        self.data_root = data_root
        self.device = device
        self.roi_size = roi_size
        self.overlap = overlap

        # (A) CNN-Backbone
        self.base_cnn = BaseCNN(model_name='resnet18', pretrained=pretrained).to(device)
        # (B) MIL aggregator (mean pooling)
        self.mil_agg = MILAggregator().to(device)
        # (C) TripletLoss
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
        # (D) Optimizer
        self.optimizer = optim.Adam(
            list(self.base_cnn.parameters()) + list(self.mil_agg.parameters()),
            lr=lr
        )

    def compute_patient_embedding(self, pid, study_yr):
        """
        Lädt ALLE Patches für (pid, study_yr) => CNN => MIL => 1 Embedding (1,512)
        (keine Grad, hier nur Inferenz)
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
        # Wir machen hier no_grad => reine Feature-Extraktion
        self.base_cnn.eval()
        self.mil_agg.eval()
        with torch.no_grad():
            for patch_t in loader:
                patch_t = patch_t.to(self.device)  # => (B,3,H,W)
                emb = self.base_cnn(patch_t)       # => (B,512)
                all_embs.append(emb)

        if len(all_embs)==0:
            return torch.zeros((1,512), device=self.device)

        patch_embs = torch.cat(all_embs, dim=0)  # => (N_patches,512)
        patient_emb = self.mil_agg(patch_embs)   # => (1,512)
        return patient_emb

    def _forward_patient(self, pid, study_yr):
        """
        Alternative, falls mit Grad rechnet (End-to-end). 
        => SinglePatientDataset + (train-mode) => RBC. 
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
        # In End-to-end Triplet => wir wollen BN/Dropout. => train
        self.base_cnn.train()
        self.mil_agg.train()
        
        for patch_t in loader:
            patch_t = patch_t.to(self.device)
            emb = self.base_cnn(patch_t)
            patch_embs.append(emb)

        if len(patch_embs)==0:
            # kreieren wir ein leeres embedding => (1,512)
            # braucht requires_grad=True, sonst kann PyTorch nicht backprop machen
            return torch.zeros((1,512), device=self.device, requires_grad=True)

        patch_embs = torch.cat(patch_embs, dim=0)
        patient_emb = self.mil_agg(patch_embs)
        return patient_emb

    def train_one_epoch(self, sampler):
        """
        sampler => Iteration über (anchor_info, pos_info, neg_info)
        In jeder Iteration => 1 Triplet => 1 Optimizer-Schritt
        """
        total_loss = 0.0
        steps = 0

        for step, (anchor_info, pos_info, neg_info) in enumerate(sampler):
            anchor_pid, anchor_sy = anchor_info['pid'], anchor_info['study_yr']
            pos_pid, pos_sy = pos_info['pid'], pos_info['study_yr']
            neg_pid, neg_sy = neg_info['pid'], neg_info['study_yr']

            # => Embeddings MIT Grad (train)
            anchor_emb = self._forward_patient(anchor_pid, anchor_sy)  # (1,512)
            pos_emb    = self._forward_patient(pos_pid, pos_sy)
            neg_emb    = self._forward_patient(neg_pid, neg_sy)

            loss = self.triplet_loss_fn(anchor_emb, pos_emb, neg_emb)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

            if step % 10 == 0:
                print(f"[Step {step}] Triplet Loss = {loss.item():.4f}")

        if steps>0:
            avg_loss = total_loss / steps
        else:
            avg_loss = 0.0
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
            sampler.reset_epoch() 

if __name__=="__main__":
    data_csv = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\nlst_subset_v5.csv"
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
        pretrained=False
    )

    trainer.train_loop(num_epochs=2, num_triplets=10)
