import sys
import os
import torch
from torch.utils.data import DataLoader

# Füge das Hauptprojektverzeichnis zum Suchpfad hinzu
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

print("Updated Python Path:", sys.path)

# Importiere die LungCTDataset-Klasse
from training.data_loader import LungCTDataset

# Definiere Pfade
data_csv = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V3\nlst_subset_v3.csv"
data_root = r"D:\thesis_robert\NLST_subset_v3_nifti_resampled_normalized"

# Dataset erstellen
dataset = LungCTDataset(data_csv=data_csv, data_root=data_root, transform=None, patch_extractor=None)

# DataLoader erstellen
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Teste den DataLoader
print("\n--- DataLoader Test ---")
try:
    for patches, label, pid in loader:
        print("Patches shape:", patches.shape)  # (2, num_patches, 3, H, W)
        print("Label shape:", label.shape)      # (2, num_labels)
        print("PID:", pid)                      # Liste mit 2 Patient-IDs
        break  # Test nur für den ersten Batch
except Exception as e:
    print(f"Fehler beim Laden der Daten: {e}")

# Zusätzliche Tests
print("\n--- Zusätzliche Tests ---")

# Überprüfe, ob alle Dateien existieren
missing_files = []
for idx in range(len(dataset)):
    pid = dataset.patient_ids[idx]
    study_yr = dataset.study_years[idx]
    file_path = os.path.join(data_root, f"pid_{pid}_study_yr_{study_yr}.nii.gz.nii.gz")
    if not os.path.exists(file_path):
        missing_files.append(file_path)

# Zusammenfassen der fehlenden Dateien
if missing_files:
    print(f"\nAnzahl fehlender Dateien: {len(missing_files)}")
    print("Beispiele für fehlende Dateien:")
    for file in missing_files[:10]:  # Zeige nur die ersten 10 fehlenden Dateien
        print(file)
    if len(missing_files) > 10:
        print("... und weitere.")
else:
    print("\nAlle Dateien vorhanden.")
