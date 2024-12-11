import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset, GridPatchDataset
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Compose, Transposed
)

class LungCTVolumeDataset(Dataset):
    """Dataset, das ganze Volumina lädt und ein Dictionary mit 'image' und 'label' zurückgibt."""
    def __init__(self, data_csv, data_root, patch_size=(64,64,3), patch_overlap=(32,32,1)):
        self.df = pd.read_csv(data_csv)
        self.data_root = data_root
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        
        self.data = []
        for i, row in self.df.iterrows():
            pid = row['pid']
            study_yr = row['study_yr']
            combination = row['combination']

            # Erzeuge Label aus der Kombination "0-0-1" -> [0,0,1]
            label_vec = [int(x) for x in combination.split('-')]

            # Erstelle den Pfad zur NIfTI-Datei
            patient_file = os.path.join(
                self.data_root,
                f"pid_{pid}_study_yr_{study_yr}.nii.gz.nii.gz"
            )
            
            # Jedes Element enthält den Pfad zum Bild und das zugehörige Label
            self.data.append({"image": patient_file, "label": label_vec})
        
        # Transforms für das Laden des gesamten Volumens
        self.volumetransforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"])  # macht aus (H,W,D) -> (1,H,W,D)
        ])

    def __getitem__(self, idx):
        d = self.volumetransforms(self.data[idx])
        # d['image'] hat jetzt Form (1, H, W, D)
        # label bleibt wie es ist
        return d

    def __len__(self):
        return len(self.data)


def get_patch_data_loader(data_csv, data_root, batch_size=2, patch_size=(64,64,3), patch_overlap=(32,32,1), num_workers=4):
    # Erzeuge das Volume-Dataset
    volume_dataset = LungCTVolumeDataset(
        data_csv=data_csv, 
        data_root=data_root, 
        patch_size=patch_size, 
        patch_overlap=patch_overlap
    )

    # GridPatchDataset zerlegt jedes Volumen in Patches
    # Jeder Eintrag von volume_dataset ist ein Dict mit {'image': Tensor(1,H,W,D), 'label': [0,0,1]...}
    # GridPatchDataset generiert daraus ein Patch pro Index.
    # spatial_size = patch_size sagt aus, dass wir in (H,W,D) Dimension patchen.
    # Da wir EnsureChannelFirstd genutzt haben ist unser Format: (1,H,W,D).
    # GridPatchDataset splittet entlang der räumlichen Dimensionen von image (ohne Channel-Dim),
    # also (H,W,D), in Patches der Größe patch_size=(64,64,3).
    patch_dataset = GridPatchDataset(
        data=volume_dataset,
        patch_size=patch_size,
        start=(0,0,0),  # Startpunkt für die Patch-Extraktion
        mode='raise',   # Raise error if image smaller than patch
        pad_mode='constant',
        pad_kwargs={'value': 0},
        pad_size=None,  # Automatisches Padding falls nötig
    )

    # Jetzt ist jedes Element aus patch_dataset ein Dictionary mit
    # 'image': Tensor(1,64,64,3) und 'label': [0,0,1]
    # Wir brauchen die Dimension (3,64,64), also permutieren wir die Achsen von (1,H,W,D_slice)
    # zu (D_slice,H,W). Hier ist D_slice=3. Wir transponieren also (1,64,64,3) → (3,64,64).
    # Dafür nutzen wir einen weiteren Transform:
    final_transforms = Compose([
        Transposed(keys=["image"], indices=(0,3,1,2))  # indices: von (C,H,W,D)->(D,H,W) wenn C=1
    ])

    # MONAI DataLoader kann direkt mit diesen Datasets umgehen
    # Wir verwenden Collation von MONAI oder PyTorch, da jetzt alle Patches gleich groß sind.
    def collate_fn(batch):
        # batch ist eine Liste von Dictionaries. Wende final_transforms auf jedes Element an.
        out_batch = []
        for b in batch:
            b = final_transforms(b)
            # b['image'] hat jetzt Form (3,64,64)
            # label in b['label'] ist unverändert, z.B. [0,0,1]
            out_batch.append((b['image'], torch.tensor(b['label'], dtype=torch.float32)))
        # out_batch ist eine Liste von (image, label)-Tuples
        images = torch.stack([x[0] for x in out_batch], dim=0) # (B,3,64,64)
        labels = torch.stack([x[1] for x in out_batch], dim=0) # (B,3)
        return images, labels

    data_loader = DataLoader(
        patch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return data_loader

if __name__ == "__main__":
    data_csv = r"C:\\Users\\rbarbir\\OneDrive - Brainlab AG\\Dipl_Arbeit\\Datensätze\\Subsets\\V3\\nlst_subset_v3.csv"
    data_root = r"D:\thesis_robert\NLST_subset_v3_nifti_resampled_normalized"
    loader = get_patch_data_loader(data_csv, data_root, batch_size=2)
    for imgs, labels in loader:
        print("Images shape:", imgs.shape)  # (2,3,64,64)
        print("Labels shape:", labels.shape) # (2,3)
        break
