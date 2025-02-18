import os
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import torch.utils.data as data
import random

def rearrange_channels_torch(patch_t):
    """
    patch_t shape: (1, H, W, D)
    Falls D == 3, wandeln wir das in (3, H, W) um.
    """
    C, H, W, D = patch_t.shape
    if D == 3:
        patch_t = patch_t.squeeze(0)    # (H, W, 3)
        patch_t = patch_t.permute(2, 0, 1)  # => (3, H, W)
    return patch_t

def random_augment_2d(patch_t, p_flip=0.5, p_rot=0.5):
    """
    Einfache 2D-Datenaugmentation für einen Patch (C,H,W).
      - Horizontal-/Vertikal-Flip (p_flip)
      - 90°-Rotation (p_rot)
    """
    # Horizontal + Vertikal flip (jeweils p_flip)
    if random.random() < p_flip:
        patch_t = torch.flip(patch_t, dims=[2])  # horizontal flip
    if random.random() < p_flip:
        patch_t = torch.flip(patch_t, dims=[1])  # vertikal flip

    # 90°-Rotationen
    if random.random() < p_rot:
        k = random.choice([0,1,2,3])  # wie oft 90° rotieren
        if k > 0:
            patch_t = torch.rot90(patch_t, k, dims=[1,2])

    return patch_t

class SinglePatientDataset(data.Dataset):
    """
    Lädt alle Patches für einen Patienten (pid, study_yr).
      1) NIfTI laden (H,W,D)
      2) optional skip_factor => volume_np[..., ::skip_factor]
      3) grid_split => 3D-Patches (C,H,W,D)
      4) rearrange_channels_torch => (3,H,W) falls D=3
      5) optional => do_augmentation => random_augment_2d

    Args:
      data_root: Pfad zum Verzeichnis mit .nii.gz
      pid, study_yr
      roi_size: z. B. (96,96,3)
      overlap:  z. B. (10,10,1)
      skip_factor: z. B. 2 => jede 2. Slice
      do_augmentation: bool => random_augment_2d
    """
    def __init__(
        self,
        data_root,
        pid,
        study_yr,
        roi_size=(96,96,3),
        overlap=(10,10,1),
        skip_factor=1,
        do_augmentation=False
    ):
        super().__init__()
        self.data_root = data_root
        self.pid = pid
        self.study_yr = study_yr
        self.roi_size = roi_size
        self.overlap = overlap
        self.skip_factor = skip_factor
        self.do_augmentation = do_augmentation

        # Pfad zur NIfTI-Datei
        fn_prefix = f"pid_{pid}_study_yr_{study_yr}"
        self.nii_path = None
        for fname in os.listdir(self.data_root):
            if fname.startswith(fn_prefix) and fname.endswith(".nii.gz"):
                self.nii_path = os.path.join(self.data_root, fname)
                break

        self.patches = []

        if os.path.exists(self.nii_path):
            self.prepare_patches()
        else:
            # Falls Datei nicht existiert => leere Patches
            self.patches = []
            print(f"Warnung: Keine passende Datei für {fn_prefix} gefunden! Die Patches-Liste ist leer.")

    def prepare_patches(self):
        # 1) NIfTI => (H,W,D)
        img = nib.load(self.nii_path)
        volume_np = img.get_fdata(dtype=np.float32)  # => (H,W,D)

        # => (1,H,W,D)
        volume_np = np.expand_dims(volume_np, axis=0)

        # 2) Skip slices => volume_np[..., ::skip_factor]
        if self.skip_factor>1:
            volume_np = volume_np[..., ::self.skip_factor]

        volume_t = torch.from_numpy(volume_np)  # (1,H,W,D')

        # 3) grid_split => Patches
        self.patches = self.grid_split(volume_t, self.roi_size, self.overlap)

    def grid_split(self, volume_t, roi_size, overlap):
        """
        Zerlegt volume_t (1,H,W,D) in 3D-Patches gem. roi_size, mit Overlap.
        """
        if volume_t.ndim == 3:
            volume_t = volume_t.unsqueeze(0)

        C, H, W, D = volume_t.shape
        stride = [
            roi_size[0] - overlap[0],
            roi_size[1] - overlap[1],
            roi_size[2] - overlap[2]
        ]

        def get_positions(size, patch_size, step):
            pos = list(range(0, size - patch_size + 1, step))
            if pos and (pos[-1] + patch_size < size):
                pos.append(size - patch_size)
            elif not pos:
                pos = [0]
            return pos

        ys = get_positions(H, roi_size[0], stride[0])
        xs = get_positions(W, roi_size[1], stride[1])
        zs = get_positions(D, roi_size[2], stride[2])

        patch_list = []
        for z in zs:
            for y in ys:
                for x in xs:
                    patch = volume_t[:, y:y+roi_size[0], x:x+roi_size[1], z:z+roi_size[2]]
                    pad_h = roi_size[0] - patch.shape[1]
                    pad_w = roi_size[1] - patch.shape[2]
                    pad_d = roi_size[2] - patch.shape[3]
                    if (pad_h>0 or pad_w>0 or pad_d>0):
                        patch = F.pad(
                            patch,
                            (0,pad_d, 0,pad_w, 0,pad_h),
                            mode='constant', value=0
                        )
                    patch_list.append(patch)
        return patch_list

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_t = self.patches[idx]  # (1,H,W,D)
        patch_t = rearrange_channels_torch(patch_t)  # => (C,H,W)

        if self.do_augmentation:
            patch_t = random_augment_2d(patch_t, p_flip=0.5, p_rot=0.5)

        return patch_t
