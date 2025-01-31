import os
import torch
import numpy as np
import nibabel as nib
import logging
import torch.nn.functional as F
import torch.utils.data as data
import random

def rearrange_channels_torch(patch_t):
    """
    patch_t shape: (1, H, W, D)
    Falls D == 3, wandeln wir das in (3, H, W).
    """
    C, H, W, D = patch_t.shape
    if D == 3:
        patch_t = patch_t.squeeze(0)      # (H,W,3)
        patch_t = patch_t.permute(2,0,1)  # (3,H,W)
    return patch_t

def random_augment_2d(patch_t, p_flip=0.5, p_rot=0.5):
    """
    Einfache 2D-Datenaugmentation für jeden Patch:
      - Horizontal-/Vertikal-Flip
      - 90°-Rotationen (Zufall)
    patch_t shape: (C,H,W)
    """
    # 1) Horizontal + Vertikal flip
    if random.random() < p_flip:
        patch_t = torch.flip(patch_t, dims=[2])  # horizontal
    if random.random() < p_flip:
        patch_t = torch.flip(patch_t, dims=[1])  # vertikal

    # 2) Random Rotation in 90°-Schritten
    if random.random() < p_rot:
        k = random.choice([0,1,2,3])  # wie oft 90° rotieren
        if k>0:
            patch_t = torch.rot90(patch_t, k, dims=[1,2])
    
    return patch_t

class SinglePatientDataset(data.Dataset):
    def __init__(
        self,
        data_root,
        pid,
        study_yr,
        roi_size=(96,96,3),
        overlap=(10,10,1),
        skip_slices=True,
        skip_factor=2,
        filter_empty_patches=False,
        min_nonzero_fraction=0.01,
        filter_uniform_patches=False,
        min_std_threshold=0.01,
        do_patch_minmax=False,
        do_augmentation=True 
    ):
        super().__init__()
        self.data_root = data_root
        self.pid = pid
        self.study_yr = study_yr
        self.roi_size = roi_size
        self.overlap = overlap

        self.skip_slices = skip_slices
        self.skip_factor = skip_factor
        self.filter_empty_patches = filter_empty_patches
        self.min_nonzero_fraction = min_nonzero_fraction
        self.filter_uniform_patches = filter_uniform_patches
        self.min_std_threshold = min_std_threshold
        self.do_patch_minmax = do_patch_minmax

        self.do_augmentation = do_augmentation

        # NIfTI-Suche
        base_prefix = f"pid_{pid}_study_yr_{study_yr}"
        self.nii_path = None
        for fname in os.listdir(self.data_root):
            if fname.startswith(base_prefix) and fname.endswith(".nii.gz"):
                cand_path = os.path.join(self.data_root, fname)
                if os.path.isfile(cand_path):
                    self.nii_path = cand_path
                    break

        if not self.nii_path:
            logging.warning(
                f"[SinglePatientDataset] Keine Datei gefunden mit Prefix='{base_prefix}'"
                f" in {self.data_root}"
            )
            self.patches = []
        else:
            self.prepare_patches()

    def prepare_patches(self):
        img = nib.load(self.nii_path)
        volume_np = img.get_fdata(dtype=np.float32)  # => (H,W,D)

        # => (1,H,W,D)
        volume_np = np.expand_dims(volume_np, axis=0)

        if self.skip_slices and self.skip_factor>1:
            volume_np = volume_np[..., ::self.skip_factor]

        volume_t = torch.from_numpy(volume_np)  # (1,H,W,D//skip_factor)

        # grid_split => patches
        patch_list = self.grid_split(volume_t, self.roi_size, self.overlap)

        # evtl. Filtern
        if self.filter_empty_patches or self.filter_uniform_patches:
            final_list = []
            for p in patch_list:
                frac = (p>1e-7).float().mean().item()
                std_val = p.std().item()

                meets_nonzero = (frac >= self.min_nonzero_fraction)
                meets_variance = (std_val >= self.min_std_threshold)

                if self.filter_empty_patches and self.filter_uniform_patches:
                    if meets_nonzero and meets_variance:
                        final_list.append(p)
                elif self.filter_empty_patches:
                    if meets_nonzero:
                        final_list.append(p)
                elif self.filter_uniform_patches:
                    if meets_variance:
                        final_list.append(p)
            self.patches = final_list
        else:
            self.patches = patch_list

    def grid_split(self, volume_t, roi_size, overlap):
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
            if pos and (pos[-1]+patch_size < size):
                pos.append(size - patch_size)
            elif not pos:
                pos = [0]
            return pos

        ys = get_positions(H, roi_size[0], stride[0])
        xs = get_positions(W, roi_size[1], stride[1])
        zs = get_positions(D, roi_size[2], stride[2])

        patches_out = []
        for z in zs:
            for y in ys:
                for x in xs:
                    patch = volume_t[:, y:y+roi_size[0], x:x+roi_size[1], z:z+roi_size[2]]
                    pad_h = roi_size[0] - patch.shape[1]
                    pad_w = roi_size[1] - patch.shape[2]
                    pad_d = roi_size[2] - patch.shape[3]
                    if (pad_h>0 or pad_w>0 or pad_d>0):
                        patch = F.pad(patch, (0,pad_d, 0,pad_w, 0,pad_h),
                                      mode='constant', value=0)
                    patches_out.append(patch)
        return patches_out

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_t = self.patches[idx]  # (1,H,W,D)
        patch_t = rearrange_channels_torch(patch_t)  # => (C,H,W)

        if self.do_patch_minmax:
            p_min = patch_t.min()
            p_max = patch_t.max()
            rng = p_max - p_min
            if rng>1e-7:
                patch_t = (patch_t - p_min)/rng
            else:
                patch_t.zero_()

        if self.do_augmentation:
            patch_t = random_augment_2d(patch_t, p_flip=0.5, p_rot=0.5)

        return patch_t
