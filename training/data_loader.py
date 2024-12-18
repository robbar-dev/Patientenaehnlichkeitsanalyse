import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Lambdad, Compose,
    RandSpatialCropSamplesd
)
import numpy as np

def rearrange_channels(x):
    # x: (1,H,W,D)
    # Ziel: (3,H,W) bei D=3
    return x.squeeze(0).permute(2,0,1)

class LungCTIterableDataset(IterableDataset):
    """
    Ein IterableDataset, das:
    - aus einer CSV (pid, study_yr, combination, ...)
    - für jeden Patienten:
      1) Das Volume lädt (NIfTI)
      2) 3D-Patches mit Overlap via grid_split erzeugt
      3) Aus jedem Patch weitere zufällige Sub-Patches via RandSpatialCropSamplesd extrahiert
      4) Jeden finalen Patch in (3,H,W)-Format bringt (kein Lambda, sondern Funktionsaufruf)
      5) Labels, pid, study_yr beibehält
      6) Jeden Patch yieldet
    """
    def __init__(self,
                 data_csv,
                 data_root,
                 roi_size=(64,64,3),
                 overlap=(32,32,1),
                 num_samples_per_patch=2):
        super().__init__()
        self.data_csv = data_csv
        self.data_root = data_root
        self.roi_size = roi_size
        self.overlap = overlap
        self.num_samples_per_patch = num_samples_per_patch

        # CSV laden
        self.df = pd.read_csv(self.data_csv)

        # Basis-Transform für ganzes Volume
        self.base_transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"])  # (H,W,D) -> (1,H,W,D)
        ])

        # RandSpatialCropSamplesd erzeugt aus einem einzelnen Patch weitere zufällige Sub-Patches
        self.rand_crop = RandSpatialCropSamplesd(
            keys=["image"],
            roi_size=self.roi_size,
            num_samples=self.num_samples_per_patch,
            random_center=True,
            random_size=False
        )

        # Lambdad mit global definierter Funktion anstelle einer Lambda-Funktion
        self.rearrange_transform = Lambdad(
            keys=["image"],
            func=rearrange_channels
        )

    def grid_split(self, volume, roi_size, overlap):
        """
        Zerlege ein 3D-Volumen in Patches mit Überschneidung.

        Args:
            volume (torch.Tensor): Das Eingabevolumen der Form (H,W,D) oder (C,H,W,D).
                                   Hier erwarten wir (H,W,D) oder (C=1,H,W,D).
            roi_size (tuple): Die Patchgröße (H, W, D).
            overlap (tuple): Überschneidungen in jeder Dimension (H, W, D).

        Returns:
            List[torch.Tensor]: Liste von 3D-Patches mit Form (1,H,W,D).
        """
        # Stelle sicher, dass C vorhanden ist
        # Nach EnsureChannelFirstd sollte volume (1,H,W,D) sein.
        # Falls volume ohne Kanal ist, füge einen hinzu.
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)  # (1,H,W,D)
        C, H, W, D = volume.shape

        patch_list = []
        stride = [roi_size[i] - overlap[i] for i in range(3)]

        for z in range(0, D - roi_size[2] + 1, stride[2]):
            for y in range(0, H - roi_size[0] + 1, stride[0]):
                for x in range(0, W - roi_size[1] + 1, stride[1]):
                    patch = volume[:, y:y+roi_size[0], x:x+roi_size[1], z:z+roi_size[2]]
                    patch_list.append(patch)

        return patch_list

    def __iter__(self):
        for i, row in self.df.iterrows():
            pid = row['pid']
            study_yr = row['study_yr']
            combination = row['combination']
            label_vec = [int(x) for x in combination.split('-')]

            patient_file = os.path.join(
                self.data_root,
                f"pid_{pid}_study_yr_{study_yr}.nii.gz.nii.gz"
            )

            if not os.path.exists(patient_file):
                continue

            # Lade komplettes Volume
            data_dict = {
                "image": patient_file,
                "label": label_vec,
                "pid": pid,
                "study_yr": study_yr
            }

            loaded = self.base_transforms(data_dict)  # { 'image': Tensor(1,H,W,D), ... }
            volume = loaded["image"]  # (1,H,W,D)

            # Wende grid_split an
            patch_list = self.grid_split(volume, self.roi_size, self.overlap)

            # Verarbeite die Patches weiter
            for patch in patch_list:
                patch_dict = {
                    "image": patch,
                    "label": label_vec,
                    "pid": pid,
                    "study_yr": study_yr
                }

                # RandSpatialCropSamplesd: erzeugt num_samples_per_patch Patches
                subpatch_list = self.rand_crop(patch_dict)  # Liste von Dicts

                for subpatch_dict in subpatch_list:
                    subpatch_dict = self.rearrange_transform(subpatch_dict)
                    yield {
                        "image": subpatch_dict["image"],
                        "label": subpatch_dict["label"],
                        "pid": subpatch_dict["pid"],
                        "study_yr": subpatch_dict["study_yr"]
                    }

def collate_fn(batch):
    images = []
    labels = []
    pids = []
    study_yrs = []
    for b in batch:
        images.append(b['image'])  # (3,64,64) Tensor
        labels.append(torch.tensor(b['label'], dtype=torch.float32))
        pids.append(b['pid'])
        study_yrs.append(b['study_yr'])

    images = torch.stack(images, dim=0)   # (B,3,64,64)
    labels = torch.stack(labels, dim=0)   # (B,3)
    return images, labels, pids, study_yrs

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
        batch_size=2,
        shuffle=False,  
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    for imgs, labels, pids, study_yrs in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        print("Images shape:", imgs.shape)   # (B,3,64,64)
        print("Labels shape:", labels.shape) # (B,3)
        print("PIDs:", pids)
        print("Study_yrs:", study_yrs)
        print("Device:", imgs.device, labels.device)
        break
