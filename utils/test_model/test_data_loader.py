import os
import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Füge das Hauptprojektverzeichnis zum Suchpfad hinzu
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# funktioniert erst nach Hauptprojektverzeichnis
from training.data_loader import LungCTIterableDataset, collate_fn

def test_data_loading(dataset):
    print("\nÜberprüfung der geladenen Daten:")
    for i, row in dataset.df.iterrows():
        print(f"Patient ID: {row['pid']}, Study Year: {row['study_yr']}, Combination: {row['combination']}")
        if i == 10:  # Zeige nur die ersten 10 Patienten
            break

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

def test_visualize_all_patches(loader):
    print("\nVisualisiere alle Patches für die ersten Patienten:")
    for imgs, labels, pids, study_yrs in loader:
        for i in range(imgs.shape[0]):
            visualize_patches_for_slices(imgs[i:i+1], pids[i], study_yrs[i])
        break  # Beende nach dem ersten Batch

def test_loader_performance(loader):
    print("\nTeste Batchgrößen und Ladezeiten:")
    for imgs, labels, pids, study_yrs in loader:
        print(f"Batch Size: {imgs.shape[0]}, Images shape: {imgs.shape}")
        print(f"Labels shape: {labels.shape}, PIDs: {pids}, Study Years: {study_yrs}")
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

    for imgs, labels, pids, study_yrs in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")
        break

if __name__ == "__main__":
    data_csv = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V3\nlst_subset_v3.csv"
    data_root = r"D:\thesis_robert\NLST_subset_v3_nifti_resampled_normalized"

    dataset = LungCTIterableDataset(
        data_csv=data_csv,
        data_root=data_root,
        roi_size=(64,64,3),
        overlap=(32,32,1),
        num_samples_per_patch=2
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loader = DataLoader(
        dataset,
        batch_size=20,
        shuffle=False,  # Shuffle entfernt, da nicht erlaubt mit IterableDataset
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Teste die verschiedenen Funktionen
    test_data_loading(dataset)
    test_visualize_all_patches(loader)
    test_loader_performance(loader)
    test_training_loop(loader, device)
