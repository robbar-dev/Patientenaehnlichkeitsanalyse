# trainer.py

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

# BaseCNN: ResNet18-Backbone, das (B,3,H,W)->(B,512) ausgibt
from model.base_cnn import BaseCNN
# MIL Aggregator: simplest (Mean) aggregator
class MILAggregator(nn.Module):
    def forward(self, patch_embs):
        # patch_embs: (N_patches, 512)
        # Return (1, 512)
        return patch_embs.mean(dim=0, keepdim=True)

# TripletSampler
from training.triplet_sampler import TripletSampler
# SinglePatientDataset
from training.single_patient_dataset import SinglePatientDataset

class TripletTrainer:
    def __init__(self, df, data_root, device='cuda', lr=1e-3, margin=1.0,
                 roi_size=(96,96,3), overlap=(10,10,1), pretrained=False):
        """
        Args:
            df: Pandas DataFrame [pid, study_yr, combination]
            data_root: Pfad zu NIfTI-Daten
            device: 'cuda' oder 'cpu'
            lr: Lernrate
            margin: Triplet-Loss margin
            roi_size, overlap: Parameter für Patch-Extraktion
            pretrained: ob ResNet18 auf ImageNet-Gewichten basiert
        """
        self.df = df
        self.data_root = data_root
        self.device = device
        self.roi_size = roi_size
        self.overlap = overlap

        # CNN: ResNet18 => (B, 512)
        self.base_cnn = BaseCNN(model_name='resnet18', pretrained=pretrained).to(device)
        # Aggregator => Mean-Pooling
        self.mil_agg = MILAggregator().to(device)

        # Triplet-Loss
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.base_cnn.parameters()) + list(self.mil_agg.parameters()),
            lr=lr
        )

    def compute_patient_embedding(self, pid, study_yr):
        """
        Lädt alle Patches dieses Patienten -> CNN -> MIL -> (1,512)
        """
        # SinglePatientDataset
        ds = SinglePatientDataset(
            data_root=self.data_root,
            pid=pid,
            study_yr=study_yr,
            roi_size=self.roi_size,
            overlap=self.overlap
        )
        loader = DataLoader(ds, batch_size=16, shuffle=False)

        patch_embs = []
        self.base_cnn.eval()    # oder train(), je nachdem
        self.mil_agg.eval()
        
        # Du KANNST in train-Phase BN/Dropout wollen; hier ggf. .train().
        # Dann solltest Du .grad() beibehalten. => Trick: Wir machen embeddings im train-mode
        # aber das kann sehr VRAM-intensiv werden. => Hier Minimallösung:
        
        with torch.no_grad():
            for patch_t in loader:
                patch_t = patch_t.to(self.device)    # (B,3,H,W)
                emb = self.base_cnn(patch_t)         # => (B,512)
                patch_embs.append(emb)
        if len(patch_embs)==0:
            # kein Patch => return zero-embedding
            return torch.zeros((1,512), device=self.device)

        patch_embs = torch.cat(patch_embs, dim=0) # => (N_patches, 512)
        patient_emb = self.mil_agg(patch_embs)    # => (1,512)
        return patient_emb

    def train_one_epoch(self, triplet_sampler):
        """
        triplet_sampler: Erzeugt (A,P,N)-Triplets => je iteration 1 Triplet -> 1 update
        """
        self.base_cnn.train()
        self.mil_agg.train()
        
        total_loss = 0.0
        for step, (anchor_info, pos_info, neg_info) in enumerate(triplet_sampler):
            # anchor_info = {pid, study_yr, combination}
            anchor_pid = anchor_info['pid']
            anchor_sy  = anchor_info['study_yr']

            pos_pid = pos_info['pid']
            pos_sy  = pos_info['study_yr']

            neg_pid = neg_info['pid']
            neg_sy  = neg_info['study_yr']

            # 1) hole embeddings
            # ACHTUNG: Hier hast Du evtl. sehr viele Patches => 
            # => Speichere Graph? => in Triplet-Learning hat man oft .no_grad() an,
            #    aber wir wollen fine-tunen => Also IM PRINZIP "with torch.enable_grad():"
            #    Minimallösung: unrolled 
            
            anchor_emb = self._forward_patient(anchor_pid, anchor_sy)
            pos_emb    = self._forward_patient(pos_pid, pos_sy)
            neg_emb    = self._forward_patient(neg_pid, neg_sy)
            
            # 2) TripletLoss => shape (1,) => scalar
            loss = self.triplet_loss_fn(anchor_emb, pos_emb, neg_emb)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            if step%10==0:
                print(f"[Step {step}] Triplet Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / (step+1)
        print(f"Epoch Loss = {avg_loss:.4f}")

    def _forward_patient(self, pid, study_yr):
        """
        Hilfsfunktion: Holt die Patches => forward => returns embedding mit grad
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
        for patch_t in loader:
            patch_t = patch_t.to(self.device)   # => (B,3,H,W)
            emb = self.base_cnn(patch_t)        # => (B,512)
            patch_embs.append(emb)

        if len(patch_embs)==0:
            # kein Patch
            return torch.zeros((1,512), device=self.device, requires_grad=True)

        patch_embs = torch.cat(patch_embs, dim=0) # => (N_patches,512)
        patient_emb = self.mil_agg(patch_embs)     # => (1,512)
        return patient_emb

    def train_loop(self, num_epochs=2, num_triplets=100):
        # Baue TripletSampler
        from training.triplet_sampler import TripletSampler
        sampler = TripletSampler(self.df, num_triplets=num_triplets, shuffle=True)

        for epoch in range(num_epochs):
            print(f"=== EPOCH {epoch+1}/{num_epochs} ===")
            self.train_one_epoch(sampler)

###############################################################################
# Minimal test main
###############################################################################
if __name__=="__main__":
    import pandas as pd
    
    # Angenommen, Du hast eine CSV -> df, die [pid, study_yr, combination] enthält
    # data_root = Pfad zu NIfTI-Dateien
    data_csv = r"PATH\TO\CSV.csv"
    data_root = r"PATH\TO\NIFTI"
    
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
