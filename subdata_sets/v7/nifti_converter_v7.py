import os
import pandas as pd
import logging
import pydicom
import nibabel as nib
import numpy as np
from glob import glob
from tqdm import tqdm

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Eingabe- und Ausgabe-Pfade
input_csv = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V7\nlst_subset_v7_normal_data.csv"
data_path = r"D:\thesis_robert\subset_v7\NLST_subset_v7_dicom_normal_unverarbeitet"
output_path_niftis = r"D:\thesis_robert\subset_v7\NLST_subset_v7_normal_unverarbeitet"
failed_nifti_output = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V7\nlst_subset_v7_failed_nifti.csv"

# CSV-Datei einlesen
logging.info("Lese die CSV-Datei ein...")
df = pd.read_csv(input_csv)

# Liste für fehlgeschlagene Konvertierungen
failed_conversions = []

# Funktion zur DICOM-zu-NIfTI-Konvertierung
def convert_dicom_to_nifti(dicom_folder, output_filename):
    try:
        dicom_files = glob(os.path.join(dicom_folder, "*.dcm"))
        if not dicom_files:
            logging.warning(f"Keine DICOM-Dateien gefunden in {dicom_folder}")
            return False
        
        dicom_data = [pydicom.dcmread(f) for f in dicom_files]
        dicom_data.sort(key=lambda x: float(x.InstanceNumber))

        pixel_array = np.stack([d.pixel_array for d in dicom_data], axis=-1)
        nifti_image = nib.Nifti1Image(pixel_array, affine=np.eye(4))
        nib.save(nifti_image, output_filename)
        return True
    except Exception as e:
        logging.error(f"Fehler bei der Konvertierung in {dicom_folder}: {e}")
        return False

# Erstelle das Ausgabe-Verzeichnis, falls nicht vorhanden
os.makedirs(output_path_niftis, exist_ok=True)

# Verarbeitung aller Patienten mit Fortschrittsanzeige
for _, row in tqdm(df.iterrows(), total=len(df), desc="Konvertiere DICOM zu NIfTI", unit="Patienten"):
    pid = str(row["pid"])
    study_yr = str(row["study_yr"])

    # Ordnername im neuen Format
    dicom_folder = os.path.join(data_path, f"pid_{pid}_study_yr_{study_yr}")

    # Ziel-Dateiname für die NIfTI-Datei
    output_filename = os.path.join(output_path_niftis, f"pid_{pid}_study_yr_{study_yr}.nii.gz")

    # Falls die NIfTI-Datei bereits existiert, überspringen
    if os.path.exists(output_filename):
        logging.info(f"Übersprungen (bereits vorhanden): {output_filename}")
        continue

    # Prüfen, ob der DICOM-Ordner existiert
    if not os.path.exists(dicom_folder):
        logging.warning(f"DICOM-Ordner nicht gefunden: {dicom_folder}")
        failed_conversions.append([pid, study_yr, "DICOM-Ordner nicht gefunden"])
        continue

    # Konvertierung durchführen
    if not convert_dicom_to_nifti(dicom_folder, output_filename):
        failed_conversions.append([pid, study_yr, "DICOM zu NIfTI Konvertierung fehlgeschlagen"])
        continue

    logging.info(f"Erfolgreich konvertiert: {output_filename}")

# Fehlgeschlagene Konvertierungen speichern
failed_df = pd.DataFrame(failed_conversions, columns=["pid", "study_yr", "Fehler"])
failed_df.to_csv(failed_nifti_output, index=False)

logging.info("Fehlgeschlagene Konvertierungen gespeichert.")
logging.info("Skript abgeschlossen.")
