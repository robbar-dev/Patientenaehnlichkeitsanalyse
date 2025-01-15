import os
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import torch.utils.data as data

def rearrange_channels_torch(patch_t):
    """
    patch_t shape: (1, H, W, D)
    Falls D == 3, wandeln wir das in (3, H, W) um
    """
    C, H, W, D = patch_t.shape
    if D == 3:
        # (1,H,W,3) => (3,H,W)
        patch_t = patch_t.squeeze(0)
        patch_t = patch_t.permute(2,0,1)
    return patch_t

class SinglePatientDataset(data.Dataset):
    """
    Lädt ALLE Patches für genau einen Patienten (pid, study_yr).
    1) NIfTI laden
    2) Slices skippen [..., ::2]
    3) grid_split => Patch-Extraktion
    4) (1,H,W,D) -> (3,H,W) falls D=3
    """

    def __init__(
        self,
        data_root,
        pid,
        study_yr,
        roi_size=(96,96,3),
        overlap=(10,10,1)
    ):
        super().__init__()
        self.data_root = data_root
        self.pid = pid
        self.study_yr = study_yr
        self.roi_size = roi_size
        self.overlap = overlap

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
        volume_np = img.get_fdata()  # float64
        volume_np = np.expand_dims(volume_np, axis=0)  # (1,H,W,D)

        # 2) skip slices =>  volume_np[..., ::2]
        volume_np = volume_np[..., ::2]

        volume_t = torch.from_numpy(volume_np).float()  # (1,H,W,D)

        # 3) grid_split => (1,H,W,D)-Patches
        self.patches = self.grid_split(volume_t, self.roi_size, self.overlap)

    def grid_split(self, volume_t, roi_size, overlap):
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
                    patch = volume_t[:, y:y+roi_size[0], x:x+roi_size[1], z:z+roi_size[2]]
                    pad_h = roi_size[0] - patch.shape[1]
                    pad_w = roi_size[1] - patch.shape[2]
                    pad_d = roi_size[2] - patch.shape[3]
                    if pad_h>0 or pad_w>0 or pad_d>0:
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
        # rearrange to (3,H,W) if D=3
        patch_t = rearrange_channels_torch(patch_t)
        return patch_t

# if __name__ == "__main__":
#     #data_root = r"D:\path\to\NIFTI"
#     # pid = 123
#     # study_yr = 2008
#     # ds = SinglePatientDataset(data_root, pid, study_yr)
#     # print("Anzahl Patches:", len(ds))
#     # from torch.utils.data import DataLoader
#     # loader = DataLoader(ds, batch_size=4, shuffle=False)
#     # for i, patch_batch in enumerate(loader):
#     #     print(f"Batch {i}, shape={patch_batch.shape}")
#     #     break