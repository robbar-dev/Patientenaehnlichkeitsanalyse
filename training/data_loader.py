import os
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import torch.utils.data as data

from monai.transforms import (
    LoadImaged, EnsureChannelFirstd
)

def rearrange_channels_torch(patch_t):
    """
    patch_t shape: (1, H, W, D)
    Falls D == 3, wandeln wir das in (3, H, W) um
    """
    # patch_t: Tensor[float32] der Form (1,H,W,D)
    C, H, W, D = patch_t.shape
    if D == 3:
        # (1,H,W,3) => (H,W,3) => (3,H,W)
        patch_t = patch_t.squeeze(0)      # => (H,W,3)
        patch_t = patch_t.permute(2,0,1)  # => (3,H,W)
    return patch_t

class SinglePatientDataset(data.Dataset):
    """
    Dieses Dataset lädt ALLE Patches für **einen** Patienten (definiert durch pid & study_yr).
    Schritte:
     1) NIfTI laden
     2) skip_slices (jeden 2. Slice)
     3) grid_split (Overlap, Patch-Extraktion, ggf. Padding)
     4) rearrange_channels (D=3 => (3,H,W))
    Danach kann man über einen DataLoader(mini-batch) drübergehen (z. B. batch_size=16),
    um die Patches in mehreren Schritten durch das CNN zu schicken.
    """

    def __init__(
        self,
        data_root,
        pid,
        study_yr,
        roi_size=(96,96,3),
        overlap=(10,10,1)
    ):
        """
        Args:
          data_root: Pfad zum Verzeichnis mit .nii.gz Dateien
          pid: Patienten-ID
          study_yr: year
          roi_size: z.B. (96,96,3)
          overlap: z.B. (10,10,1)
        """
        super().__init__()
        self.data_root = data_root
        self.pid = pid
        self.study_yr = study_yr

        self.roi_size = roi_size
        self.overlap = overlap

        # Pfad zum NIfTI
        self.nii_path = os.path.join(
            self.data_root,
            f"pid_{pid}_study_yr_{study_yr}.nii.gz.nii.gz"
        )

        # Liste von 4D-Tensoren, jeder (1,H,W,D)
        self.patches = []
        if os.path.exists(self.nii_path):
            self.prepare_patches()

    def prepare_patches(self):
        """
        Lädt das Volume, wendet Skip Slices, Grid Split an
        und speichert alle extrahierten Patches in self.patches.
        """
        # 1) NIfTI laden
        img = nib.load(self.nii_path)
        volume_np = img.get_fdata()  # => (H,W,D), float64

        # 2) Expand Channel => (1,H,W,D)
        volume_np = np.expand_dims(volume_np, axis=0)  # (1,H,W,D)

        # 3) Skip Slices => [::2]
        volume_np = volume_np[..., ::2]  # => (1,H,W,D//2)

        # 4) In Torch konvertieren
        volume_t = torch.from_numpy(volume_np).float()  # => (1,H,W,D)

        # 5) grid_split => Liste von (1,H,W,D) Patches
        self.patches = self.grid_split(volume_t, self.roi_size, self.overlap)

    def grid_split(self, volume_t, roi_size, overlap):
        """
        1:1 deine Logik aus dem IterableDataset, aber auf dem Volume-Tensor statt yield.
        """
        if volume_t.ndim == 3:
            volume_t = volume_t.unsqueeze(0)  # => (1,H,W,D)

        C, H, W, D = volume_t.shape
        stride = [roi_size[i] - overlap[i] for i in range(3)]

        def get_positions(size, patch_size, step):
            positions = list(range(0, size - patch_size + 1, step))
            if positions and (positions[-1] + patch_size < size):
                positions.append(size - patch_size)
            elif not positions:
                positions = [0]
            return positions

        ys = get_positions(H, roi_size[0], stride[0])
        xs = get_positions(W, roi_size[1], stride[1])
        zs = get_positions(D, roi_size[2], stride[2])

        patch_list = []
        for z in zs:
            for y in ys:
                for x in xs:
                    patch = volume_t[:, y:y+roi_size[0], x:x+roi_size[1], z:z+roi_size[2]]
                    # Padding falls Patch kleiner als roi_size
                    pad_h = roi_size[0] - patch.shape[1]
                    pad_w = roi_size[1] - patch.shape[2]
                    pad_d = roi_size[2] - patch.shape[3]
                    if pad_h>0 or pad_w>0 or pad_d>0:
                        patch = F.pad(
                            patch,
                            (0,pad_d, 0,pad_w, 0,pad_h),  # (D_before, D_after, W_before, W_after, H_before, H_after)
                            mode='constant', value=0
                        )
                    patch_list.append(patch)
        return patch_list

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        """
        Gibt den idx-ten Patch zurück, konvertiert von (1,H,W,D) -> (3,H,W) falls D=3
        """
        patch_t = self.patches[idx]  # (1,H,W,D)
        # rearrange channels
        patch_t = rearrange_channels_torch(patch_t)  # => evtl. (3,H,W)
        return patch_t


if __name__ == "__main__":
    """
    Beispiel: Du willst alle Patches für (pid=123, study_yr=2008) extrahieren,
    dann mit DataLoader in mini-batches verarbeiten.
    """
    # Dummy Pfade
    data_root = r"D:\thesis_robert\NLST_subset_v3_nifti_resampled_normalized"
    pid = 123
    study_yr = 2008

    ds = SinglePatientDataset(
        data_root=data_root,
        pid=pid,
        study_yr=study_yr,
        roi_size=(96,96,3),
        overlap=(10,10,1)
    )

    print("Anzahl Patches:", len(ds))

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    for batch_idx, patch_batch in enumerate(loader):
        # patch_batch shape: (B,3,H,W) oder (B, C,H,W), je nach D
        print(f"Batch {batch_idx}: {patch_batch.shape}")
        # hier würdest du dein CNN auf patch_batch aufrufen
        break
