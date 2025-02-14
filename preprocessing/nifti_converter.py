import os
import pandas as pd
import logging
import pydicom
import nibabel as nib
import numpy as np
from glob import glob
from tqdm import tqdm

# Logging konfigurieren
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Eingabe- und Ausgabe-Pfade
input_csv = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V3\nlst_subset_v3.csv"
data_path = r"M:\public_data\tcia_ml\nlst\ct"
failed_nifti_output = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V3\subset_v3_failed_nifti.csv"
output_path_niftidatein = r"D:\thesis_robert\NLST_subset_v3_series_nifti_unverarbeitet"

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
        logging.error(f"Fehler bei der Konvertierung: {e}")
        return False

# Verarbeitung aller Patienten mit Ladebalken
for _, row in tqdm(df.iterrows(), total=len(df), desc="Verarbeite Patienten", unit="Patienten"):
    pid = str(row["pid"])
    study_yr = str(row["study_yr"])
    
    patient_folder = os.path.join(data_path, pid)
    if not os.path.exists(patient_folder):
        logging.warning(f"Patientenordner nicht gefunden: {patient_folder}")
        failed_conversions.append([pid, study_yr, "Patientenordner nicht gefunden"])
        continue
    
    study_folder = None
    for folder in os.listdir(patient_folder):
        if folder.startswith("01-02-") and folder.split("-")[2] == str(1999 + int(study_yr)):
            study_folder = os.path.join(patient_folder, folder)
            break
    
    if not study_folder:
        logging.warning(f"Kein passender Study-Year-Ordner gefunden für PID {pid}, Study_Year {study_yr}")
        failed_conversions.append([pid, study_yr, "Study-Year-Ordner nicht gefunden"])
        continue
    
    # Rekursiv den obersten Unterordner mit DICOM-Dateien finden
    dicom_folder = study_folder
    while os.path.isdir(dicom_folder):
        subfolders = sorted(os.listdir(dicom_folder))
        if not subfolders:
            break
        dicom_folder = os.path.join(dicom_folder, subfolders[0])
        if glob(os.path.join(dicom_folder, "*.dcm")):
            break
    
    # Ziel-Dateiname
    output_filename = os.path.join(output_path_niftidatein, f"pid_{pid}_study_yr_{study_yr}.nii.gz")
    
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
