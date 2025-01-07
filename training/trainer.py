import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Füge das Hauptprojektverzeichnis zum Suchpfad hinzu (sofern nötig)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 1) Importiere Deinen DataLoader & Dataset
from training.data_loader import LungCTIterableDataset, collate_fn

# 2) Importiere Deinen ResNet18-Feature-Extraktor
from model.base_cnn import BaseCNN

###############################################################################
# 1) Ein sehr einfaches Aggregationsmodul (Mean Pooling über Batch)
###############################################################################
class SimpleBatchAggregator(nn.Module):
    """
    Für Sprint A: wir mitteln die Embeddings aller Patches im Batch
    Einfacher Ersatz für richtige MIL-Logik (die wäre patientenweise).
    """
    def forward(self, patch_embeddings):
        # patch_embeddings: (B, 512) bei ResNet18
        patient_emb = patch_embeddings.mean(dim=0, keepdim=True)  # => (1, 512)
        return patient_emb

###############################################################################
# 2) Beispielhafter Trainer
###############################################################################
class Trainer:
    def __init__(self,
                 data_csv,
                 data_root,
                 batch_size=3,
                 lr=1e-3,
                 device="cuda",
                 pretrained=False):
        """
        Args:
          data_csv: CSV-Datei mit den Patienten (für Sprint A: Minidatensatz)
          data_root: Pfad zu den .nii.gz-Dateien
          batch_size: Anzahl Patches pro Batch
          lr: Lernrate
          device: 'cuda' oder 'cpu'
          pretrained: Ob das ResNet18 ImageNet-Gewichte lädt
        """
        self.device = device
        
        # (1) Model: ResNet18 als Feature-Extraktor => 512-dim Output
        self.base_cnn = BaseCNN(model_name='resnet18', pretrained=pretrained).to(device)
        
        # (2) Aggregator: Mittelwert über alle Patches im Batch
        self.aggregator = SimpleBatchAggregator().to(device)
        
        # (3) Optimizer
        self.optimizer = optim.Adam(
            list(self.base_cnn.parameters()) + list(self.aggregator.parameters()),
            lr=lr
        )

        # (4) Dummy-Loss: MSE mit 0-Vektor (512-dim)
        self.criterion = nn.MSELoss()

        # (5) Dataset & DataLoader
        dataset = LungCTIterableDataset(
            data_csv=data_csv,
            data_root=data_root,
            roi_size=(96,96,3),
            overlap=(10,10,1)
        )
        
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  
            num_workers=0,  # Debug
            pin_memory=True,
            collate_fn=collate_fn
        )

    def train_one_epoch(self):
        self.base_cnn.train()
        self.aggregator.train()
        
        total_loss = 0.0
        for step, (imgs, labels, pids, study_yrs) in enumerate(self.loader):
            # imgs shape: (B, 3, H=96, W=96)
            imgs = imgs.to(self.device)
            
            # 1) CNN Forward: => (B, 512)
            patch_embs = self.base_cnn(imgs)
            
            # 2) Aggregation => (1, 512)
            patient_emb = self.aggregator(patch_embs)
            
            # 3) Dummy-Loss: MSE gegen Null
            target = torch.zeros_like(patient_emb)
            loss = self.criterion(patient_emb, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            if step % 10 == 0:
                print(f"[Step {step}] Loss = {loss.item():.4f}")

        avg_loss = total_loss / (step + 1)
        print(f"Epoch loss = {avg_loss:.4f}")

    def train_loop(self, num_epochs=1):
        for epoch in range(num_epochs):
            print(f"=== EPOCH {epoch+1}/{num_epochs} ===")
            self.train_one_epoch()

###############################################################################
# 3) Main
###############################################################################
if __name__ == "__main__":
    # Pfade anpassen: Du kannst Dir einen kleinen Satz (z.B. 12 Patienten) anlegen.
    data_csv = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\nlst_subset_v5.csv"
    data_root = r"D:\thesis_robert\NLST_subset_v5_nifti_3mm_Voxel"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = Trainer(
        data_csv=data_csv,
        data_root=data_root,
        batch_size=3,
        lr=1e-3,
        device=device,
        pretrained=False
    )

    trainer.train_loop(num_epochs=2)
