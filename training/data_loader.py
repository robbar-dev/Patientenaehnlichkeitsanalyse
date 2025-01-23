import os
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import torch.utils.data as data

def rearrange_channels_torch(patch_t):
    """
    patch_t shape: (1, H, W, D)
    Falls D == 3, wandeln wir das in (3, H, W) um.
    """
    C, H, W, D = patch_t.shape
    if D == 3:
        # (1,H,W,3) => (3,H,W)
        patch_t = patch_t.squeeze(0)
        patch_t = patch_t.permute(2, 0, 1)
    return patch_t

class SinglePatientDataset(data.Dataset):
    """
    Lädt ALLE Patches für genau einen Patienten (pid, study_yr).
    1) NIfTI laden
    2) Slices skippen => volume_np[..., ::2]
    3) grid_split => Patch-Extraktion
    4) (1,H,W,D) -> (3,H,W) falls D=3
    5) Optional: Filtern leerer Patches
    6) Im __getitem__ => Min-Max Normalisierung
    """

    def __init__(
        self,
        data_root,
        pid,
        study_yr,
        roi_size=(96, 96, 3),
        overlap=(10, 10, 1),
        filter_empty_patches=True,
        min_nonzero_fraction=0.01
    ):
        """
        Args:
            data_root (str): Pfad zum Verzeichnis mit den .nii.gz Dateien
            pid (str): Patienten-ID
            study_yr (str): Jahr
            roi_size (tuple): z.B. (96,96,3)
            overlap (tuple): Overlap (H,W,D)
            filter_empty_patches (bool): Wenn True => wir filtern Patches ohne Variation
            min_nonzero_fraction (float): z.B. 0.01 => Mind. 1% Non-Zero, sonst Patch skip
        """
        super().__init__()
        self.data_root = data_root
        self.pid = pid
        self.study_yr = study_yr
        self.roi_size = roi_size
        self.overlap = overlap

        self.filter_empty_patches = filter_empty_patches
        self.min_nonzero_fraction = min_nonzero_fraction

        # Pfad zur .nii.gz
        self.nii_path = os.path.join(
            self.data_root,
            f"pid_{pid}_study_yr_{study_yr}.nii.gz.nii.gz"
        )
        self.patches = []
        if os.path.exists(self.nii_path):
            self.prepare_patches()

    def prepare_patches(self):
        # 1) NIfTI laden => (H,W,D)
        img = nib.load(self.nii_path)
        volume_np = img.get_fdata()  # float64 => HU oder [0..1] ...
        volume_np = np.expand_dims(volume_np, axis=0)  # => (1,H,W,D)

        # 2) skip slices => volume_np[..., ::2]
        volume_np = volume_np[..., ::2]

        volume_t = torch.from_numpy(volume_np).float()  # => (1,H,W,D)

        # 3) grid_split => Liste von (1,H,W,D)-Tensors
        all_patches = self.grid_split(volume_t, self.roi_size, self.overlap)

        # 4) (Optional) Filtern leerer/nahezu leerer Patches
        if self.filter_empty_patches:
            filtered = []
            for p in all_patches:
                # z.B. check fraction of non-zero
                nonzero_frac = (p > 0).float().mean().item()
                if nonzero_frac >= self.min_nonzero_fraction:
                    filtered.append(p)
            self.patches = filtered
        else:
            self.patches = all_patches

    def grid_split(self, volume_t, roi_size, overlap):
        """
        Zerlege das Volume in 3D-Patches mit Overlap.
        volume_t shape: (1,H,W,D)
        """
        if volume_t.ndim == 3:
            volume_t = volume_t.unsqueeze(0)
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
                    patch = volume_t[
                        :,
                        y:y+roi_size[0],
                        x:x+roi_size[1],
                        z:z+roi_size[2]
                    ]
                    # Padding falls Patch kleiner als roi_size
                    pad_h = roi_size[0] - patch.shape[1]
                    pad_w = roi_size[1] - patch.shape[2]
                    pad_d = roi_size[2] - patch.shape[3]
                    if pad_h>0 or pad_w>0 or pad_d>0:
                        patch = F.pad(
                            patch,
                            (0, pad_d, 0, pad_w, 0, pad_h),  # (D_before,D_after,W_before,W_after,H_before,H_after)
                            mode='constant', value=0
                        )
                    patch_list.append(patch)
        return patch_list

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        """
        Holt den idx-ten Patch => (1,H,W,D)
        => rearrange_channels_torch => (D,H,W) oder (3,H,W)
        => min-max-normalisierung
        => return
        """
        patch_t = self.patches[idx]  # (1,H,W,D)
        # => (3,H,W) falls D=3
        patch_t = rearrange_channels_torch(patch_t)

        # => min-max normalisieren
        p_min = patch_t.min()
        p_max = patch_t.max()
        rng = p_max - p_min
        if rng > 1e-7:
            patch_t = (patch_t - p_min) / rng
        else:
            # alles gleich => setze patch auf 0
            patch_t.zero_()

        return patch_t
