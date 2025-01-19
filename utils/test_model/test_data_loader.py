import os
import sys
import torch
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader

# Füge das Hauptprojektverzeichnis zum Suchpfad hinzu
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importiere den neuen DataLoader
from training.data_loader import SinglePatientDataset

def test_data_loading(dataset):
    print("\nÜberprüfung der geladenen Patches:")
    print(f"Anzahl der Patches: {len(dataset)}")

def visualize_patches_for_slices(patches, pid, study_yr):
    """
    Visualisiere alle Patches von den ersten drei Slices.
    """
    print(f"\nVisualisiere Patches von Patient ID: {pid}, Study Year: {study_yr}")
    num_patches = patches.shape[0]  # Gesamtanzahl der Patches
    plt.figure(figsize=(15, 5))

    for idx in range(num_patches):
        patch = patches[idx]
        for slice_idx in range(patch.shape[0]):  # 3 Slices (Channels)
            plt.subplot(num_patches, 3, idx * 3 + slice_idx + 1)
            plt.imshow(patch[slice_idx].cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.title(f"Slice {slice_idx + 1}")

    plt.suptitle(f"Patient ID: {pid}, Study Year: {study_yr}")
    plt.tight_layout()
    plt.show()

def test_visualize_all_patches(loader, pid, study_yr):
    print("\nVisualisiere alle Patches für die ersten Patienten:")
    for patches in loader:
        visualize_patches_for_slices(patches, pid, study_yr)
        break  # Beende nach dem ersten Batch

def test_loader_performance(loader):
    print("\nTeste Batchgrößen und Ladezeiten:")
    for patches in loader:
        print(f"Batch Size: {patches.shape[0]}, Patches shape: {patches.shape}")
        break

def test_training_loop(loader, device):
    print("\nStarte einen einfachen Trainingsloop-Test:")
    import torch.nn as nn
    import torch.optim as optim

    class DummyModel(nn.Module):
        def __init__(self, input_channels=3, num_classes=3):
            super().__init__()
            self.conv = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(16, num_classes)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)  # Flatten
            return self.fc(x)

    model = DummyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    for patches in loader:
        patches = patches.to(device)

        optimizer.zero_grad()
        outputs = model(patches)
        labels = torch.zeros(outputs.shape).to(device)  # Dummy-Labels
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")
        break

if __name__ == "__main__":
    data_root = r"D:\thesis_robert\NLST_subset_v5_nifti_1_5mm_Voxel"
    pid = 216500
    study_yr = 0

    # Initialisiere das Dataset mit dem neuen Dataloader
    dataset = SinglePatientDataset(
        data_root=data_root,
        pid=pid,
        study_yr=study_yr,
        roi_size=(128, 128, 3),
        overlap=(16, 16, 1),
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loader = DataLoader(
        dataset,
        batch_size=6,
        shuffle=False,  # Shuffle entfernt, da nicht erlaubt mit IterableDataset
        num_workers=4,
        pin_memory=True,
    )

    # Teste die verschiedenen Funktionen
    test_data_loading(dataset)
    test_visualize_all_patches(loader, pid, study_yr)
    test_loader_performance(loader)
    test_training_loop(loader, device)
