import os
import torch
import numpy as np
import nibabel as nib
import logging
import torch.nn.functional as F
import torch.utils.data as data

def rearrange_channels_torch(patch_t):
    """
    patch_t shape: (1, H, W, D)
    Falls D == 3, wandeln wir das in (3, H, W).

    -> (1,H,W,3) => remove channel=1 => (H,W,3) => permute(2,0,1) => (3,H,W)
    """
    C, H, W, D = patch_t.shape
    if D == 3:
        patch_t = patch_t.squeeze(0)      # (H,W,3)
        patch_t = patch_t.permute(2,0,1)  # (3,H,W)
    return patch_t

class SinglePatientDataset(data.Dataset):
    """
    Ein Dataset, das aus EINEM NIfTI (pid, study_yr) 3D-Volume alle Patches extrahiert.
    Schritte:
     1) NIfTI laden
     2) Optional: Slices skippen (e.g. volume[..., ::2])
     3) Patches via grid_split
     4) Optional: Filtern von Patches, die (fast) leer sind
        - und/oder die (fast) keine Varianz aufweisen
     5) Im __getitem__ -> rearrange => (3,H,W) + minmax-normalisierung
    """

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
        do_patch_minmax=False
    ):
        """
        Args:
            data_root (str): Pfad zum Ordner mit .nii.gz
            pid (str): Patienten-ID
            study_yr (str): Study-Year
            roi_size (tuple): z.B. (96,96,3)
            overlap (tuple): (10,10,1)
            skip_slices (bool): ob wir volume[..., ::skip_factor] machen
            skip_factor (int): standard=2 => jede 2. Slice
            filter_empty_patches (bool): ob wir Patches, die ~leer sind, verwerfen
            min_nonzero_fraction (float): z.B. 0.01 => mind. 1% Non-Zero im Patch
            filter_uniform_patches (bool): ob wir Patches mit kaum Varianz filtern
            min_std_threshold (float): z.B. 0.01 => mind. diese Std-Abweichung
            do_patch_minmax (bool): ob wir pro Patch min–max normalisieren
        """
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

        # Pfad zur .nii.gz finden
        base_prefix = f"pid_{pid}_study_yr_{study_yr}"
        self.nii_path = None

        for fname in os.listdir(self.data_root):
            # 1) Prüfe: fname startet mit base_prefix?
            if fname.startswith(base_prefix):
                # 2) Prüfe: fname endet mit '.nii.gz'?
                if fname.endswith(".nii.gz"):
                    cand_path = os.path.join(self.data_root, fname)
                    if os.path.isfile(cand_path):
                        self.nii_path = cand_path
                        break

        if not self.nii_path:
            logging.warning(f"[SinglePatientDataset] Keine Datei gefunden mit Prefix='{base_prefix}' in {self.data_root}")
            self.patches = []
        else:
            self.prepare_patches()

    def prepare_patches(self):
        # (1) NIfTI laden => shape (H,W,D) i.d.R.
        img = nib.load(self.nii_path)
        volume_np = img.get_fdata()  # float64
        # => (H,W,D)

        # => expand zu (1,H,W,D)
        volume_np = np.expand_dims(volume_np, axis=0)

        # (2) optional slice skip
        if self.skip_slices and self.skip_factor>1:
            volume_np = volume_np[..., ::self.skip_factor]

        volume_t = torch.from_numpy(volume_np).float()  # => (1,H,W,D//skip_factor)

        # (3) grid_split => Patches
        patch_list = self.grid_split(volume_t, self.roi_size, self.overlap)

        # (4) Filter leerer oder „uniformer“ Patches
        if self.filter_empty_patches or self.filter_uniform_patches:
            final_list = []
            for p in patch_list:
                # fraction of nonzero
                frac = (p > 1e-7).float().mean().item()
                # std-Abweichung
                std_val = p.std().item()

                # Bedingung 1: Patch ist nicht (fast) leer
                meets_nonzero = (frac >= self.min_nonzero_fraction)
                # Bedingung 2: Patch hat genug Varianz
                meets_variance = (std_val >= self.min_std_threshold)

                # Wenn beide Filter aktiv sind, muss der Patch beide Bedingungen erfüllen.
                # Wenn nur ein Filter aktiv ist, reicht die jeweilige Bedingung.
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
        """
        Zerlegt (1,H,W,D) in 3D-Patches mit Overlap.
        """
        if volume_t.ndim == 3:
            volume_t = volume_t.unsqueeze(0)  # => (1,H,W,D)

        C, H, W, D = volume_t.shape
        stride = [
            roi_size[0] - overlap[0],
            roi_size[1] - overlap[1],
            roi_size[2] - overlap[2]
        ]

        def get_positions(size, patch_size, step):
            positions = list(range(0, size - patch_size + 1, step))
            if positions and (positions[-1] + patch_size < size):
                positions.append(size - patch_size)
            elif not positions:
                # wenn gar nix passt => Position=0
                positions = [0]
            return positions

        ys = get_positions(H, roi_size[0], stride[0])
        xs = get_positions(W, roi_size[1], stride[1])
        zs = get_positions(D, roi_size[2], stride[2])

        patches_out = []
        for z in zs:
            for y in ys:
                for x in xs:
                    patch = volume_t[
                        :,
                        y:y+roi_size[0],
                        x:x+roi_size[1],
                        z:z+roi_size[2]
                    ]
                    # padding, falls randseitig das Patch nicht ganz passt
                    pad_h = roi_size[0] - patch.shape[1]
                    pad_w = roi_size[1] - patch.shape[2]
                    pad_d = roi_size[2] - patch.shape[3]
                    if (pad_h>0 or pad_w>0 or pad_d>0):
                        patch = F.pad(
                            patch,
                            (0,pad_d, 0,pad_w, 0,pad_h),
                            mode='constant', value=0
                        )
                    patches_out.append(patch)
        return patches_out

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        """
        1) hole self.patches[idx] => (1,H,W,D)
        2) rearrange => (3,H,W) wenn D=3
        3) optional min-max => [0..1]
        """
        patch_t = self.patches[idx]  # shape (1,H,W,D)
        patch_t = rearrange_channels_torch(patch_t)

        if self.do_patch_minmax:
            p_min = patch_t.min()
            p_max = patch_t.max()
            rng = p_max - p_min
            if rng > 1e-7:
                patch_t = (patch_t - p_min)/rng
            else:
                patch_t.zero_()

        return patch_t
