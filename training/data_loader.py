import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Lambdad, Compose
    #RandSpatialCropSamplesd  # auskommentiert, da derzeit nicht genutzt
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
      2) 3D-Patches mit Overlap via grid_split erzeugt (mit Padding am Rand)
      3) Wenn gewünscht, könnte RandSpatialCropSamplesd eingesetzt werden (aktuell auskommentiert)
      4) Jeden finalen Patch in (3,H,W)-Format bringt
      5) Labels, pid, study_yr beibehält
      6) Jeden Patch yieldet
    """
    def __init__(self,
                 data_csv,
                 data_root,
                 roi_size=(96,96,3),
                 overlap=(10,10,1),
                 #num_samples_per_patch=2  # aktuell nicht genutzt
                 ):
        super().__init__()
        self.data_csv = data_csv
        self.data_root = data_root
        self.roi_size = roi_size
        self.overlap = overlap

        self.df = pd.read_csv(self.data_csv)

        # Basis-Transform für ganzes Volume
        self.base_transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"])  # (H,W,D) -> (1,H,W,D)
        ])

        # RandSpatialCropSamplesd auskommentiert, um Patch-Anzahl gering zu halten
        # self.rand_crop = RandSpatialCropSamplesd(
        #     keys=["image"],
        #     roi_size=self.roi_size,
        #     num_samples=self.num_samples_per_patch,
        #     random_center=True,
        #     random_size=False
        # )

        self.rearrange_transform = Lambdad(
            keys=["image"],
            func=rearrange_channels
        )

    def grid_split(self, volume, roi_size, overlap):
        """
        Zerlege ein 3D-Volumen in Patches unter Berücksichtigung von Overlap und
        füge bei Bedarf am Rand zusätzliche Patches mit Padding hinzu, um das gesamte
        Volumen abzudecken.

        Args:
            volume (torch.Tensor): Eingabevolumen (1,H,W,D).
            roi_size (tuple): Patchgröße (H,W,D).
            overlap (tuple): Overlap in jeder Dimension (H,W,D).

        Returns:
            List[torch.Tensor]: Liste von gepaddeten 3D-Patches (1,H,W,D).
        """
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)  # (1,H,W,D)
        C, H, W, D = volume.shape

        stride = [roi_size[i] - overlap[i] for i in range(3)]

        def get_positions(size, patch_size, step):
            positions = list(range(0, size - patch_size + 1, step))
            # Prüfe, ob noch ein Rest übrig bleibt, der kein ganzes Patch füllt
            if positions and (positions[-1] + patch_size < size):
                positions.append(size - patch_size)
            elif not positions:
                # Falls gar kein Patch passt, trotzdem ein Patch am Anfang ansetzen
                positions = [0]
            return positions

        ys = get_positions(H, roi_size[0], stride[0])
        xs = get_positions(W, roi_size[1], stride[1])
        zs = get_positions(D, roi_size[2], stride[2])

        patch_list = []
        for z in zs:
            for y in ys:
                for x in xs:
                    patch = volume[:, y:y+roi_size[0], x:x+roi_size[1], z:z+roi_size[2]]
                    # Prüfe, ob Patch kleiner als roi_size ist:
                    pad_h = roi_size[0] - patch.shape[1]
                    pad_w = roi_size[1] - patch.shape[2]
                    pad_d = roi_size[2] - patch.shape[3]

                    if pad_h > 0 or pad_w > 0 or pad_d > 0:
                        # Padding in Reihenfolge: (D_before,D_after,W_before,W_after,H_before,H_after)
                        # Wir müssen allerdings am Ende padden, also alle "afters" nutzen.
                        patch = torch.nn.functional.pad(
                            patch,
                            (0, pad_d, 0, pad_w, 0, pad_h),  # Reihenfolge: (D,W,H)
                            mode='constant', value=0
                        )

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

            data_dict = {
                "image": patient_file,
                "label": label_vec,
                "pid": pid,
                "study_yr": study_yr
            }

            loaded = self.base_transforms(data_dict)  # { 'image': Tensor(1,H,W,D), ... }
            volume = loaded["image"]  # (1,H,W,D)

            patch_list = self.grid_split(volume, self.roi_size, self.overlap)

            for patch in patch_list:
                patch_dict = {
                    "image": patch,
                    "label": label_vec,
                    "pid": pid,
                    "study_yr": study_yr
                }

                # Ohne RandSpatialCropSamplesd direkt rearrange_transform anwenden
                patch_dict = self.rearrange_transform(patch_dict)

                yield {
                    "image": patch_dict["image"],
                    "label": patch_dict["label"],
                    "pid": patch_dict["pid"],
                    "study_yr": patch_dict["study_yr"]
                }

def collate_fn(batch):
    images = []
    labels = []
    pids = []
    study_yrs = []
    for b in batch:
        images.append(b['image'])  # (3,H,W) Tensor
        labels.append(torch.tensor(b['label'], dtype=torch.float32))
        pids.append(b['pid'])
        study_yrs.append(b['study_yr'])

    images = torch.stack(images, dim=0)   # (B,3,H,W)
    labels = torch.stack(labels, dim=0)   # (B,3)
    return images, labels, pids, study_yrs

if __name__ == "__main__":
    data_csv = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V3\nlst_subset_v3.csv"
    data_root = r"D:\thesis_robert\NLST_subset_v3_nifti_resampled_normalized"

    dataset = LungCTIterableDataset(
        data_csv=data_csv,
        data_root=data_root,
        roi_size=(96,96,3),
        overlap=(10,10,1)
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
        print("Images shape:", imgs.shape)   # (B,3,96,96)
        print("Labels shape:", labels.shape) # (B,3)
        print("PIDs:", pids)
        print("Study_yrs:", study_yrs)
        print("Device:", imgs.device, labels.device)
        break
