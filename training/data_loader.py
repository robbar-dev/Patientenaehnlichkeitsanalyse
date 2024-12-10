import torch
from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
import pandas as pd

class LungCTDataset(Dataset):
    def __init__(self, 
                 data_csv,            # Pfad zur CSV-Datei mit pid, study_yr, combination, dicom_path
                 data_root,           # Verzeichnis mit den vorverarbeiteten CT-Daten (NIfTI)
                 transform=None,
                 patch_extractor=None # Funktion oder Klasse für die Patch-Extraktion
                 ):
        """
        Args:
          data_csv (str): Pfad zur CSV-Datei.
          data_root (str): Verzeichnis der normalisierten NIfTI-Dateien.
          transform (callable, optional): Optionaler Transform auf die Patches.
          patch_extractor (callable, optional): Funktion/Klasse, die aus 3D-Volumen Patches extrahiert.
        """
        self.data_csv = data_csv
        self.data_root = data_root
        self.transform = transform
        self.patch_extractor = patch_extractor

        # CSV laden
        self.df = pd.read_csv(self.data_csv)
        
        # Extrahiere Patienten-IDs, Study Year und Kombinationen
        self.patient_ids = self.df['pid'].tolist()
        self.study_years = self.df['study_yr'].tolist()
        self.combinations = self.df['combination'].tolist()  # z.B. "0-0-1"

        # Wandle die Kombinationen in numerische Labels um "0-0-1" -> [0,0,1]
        self.labels = []
        for comb in self.combinations:
            parts = comb.split('-')
            label_vec = [int(x) for x in parts]
            self.labels.append(label_vec)
        
        self.labels = np.array(self.labels)  # shape (num_patients, 3)
        
        # dicom_path aus CSV falls nötig
        # self.dicom_paths = self.df['dicom_path'].tolist()
        
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        study_yr = self.study_years[idx]
        label = self.labels[idx]  # z.B. array([0,0,1])
        
        # Erstelle den Dateinamen basierend auf pid und study_yr:
        patient_file = os.path.join(
            self.data_root, 
            f"pid_{pid}_study_yr_{study_yr}.nii.gz.nii.gz"
        )
        
        if not os.path.exists(patient_file):
            raise FileNotFoundError(f"Datei nicht gefunden: {patient_file}")

        # CT-Volumen laden
        ct_data = nib.load(patient_file).get_fdata()  # shape (H, W, D) oder (D, H, W)
        
        # Patches extrahieren
        if self.patch_extractor is not None:
            patches = self.patch_extractor(ct_data)
        else:
            # Dummy: Patch aus der Mitte
            ct_tensor = torch.from_numpy(ct_data).float()  
            # Wir gehen davon aus, ct_data ist (H,W,D):
            d = ct_tensor.shape[-1]
            mid = d // 2
            patch = ct_tensor[..., mid-1:mid+2]  # (H,W,3)
            patch = patch.permute(2,0,1)  # (3,H,W)
            patches = patch.unsqueeze(0)   # (1,3,H,W)
        
        if self.transform is not None:
            patches = self.transform(patches)
        
        return patches, torch.tensor(label, dtype=torch.float32), pid
