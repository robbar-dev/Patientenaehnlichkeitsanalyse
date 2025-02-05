import os
import pandas as pd
import logging
import pydicom
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm
import random

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Eingabe- und Ausgabe-Pfade
csv_file = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V6\subset_v6.csv"
not_normal_file = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V7\not_normal.xlsx"
dicom_input_dir = r"M:\public_data\tcia_ml\nlst\ct"
output_path_slices = r"D:\thesis_robert\subset_v7\NLST_subset_v7_dicom_normal_unverarbeitet"

# Sicherstellen, dass das Ausgabe-Verzeichnis existiert
os.makedirs(output_path_slices, exist_ok=True)

# Anzahl der zu ersetzenden Patienten (entspricht den "not normal" Patienten)
df_not_normal = pd.read_excel(not_normal_file, header=None)
df_not_normal.columns = ["filename"]
df_not_normal["pid_study_yr"] = df_not_normal["filename"].str.replace(".nii.gz.nii", "", regex=False)

num_to_replace = len(df_not_normal)

# CSV-Datei mit gesunden Patienten einlesen
df = pd.read_csv(csv_file, sep=";")

# Gesunde Patienten auswählen, aber die aus der "not_normal"-Liste ausschließen
healthy_patients = df[df["combination"] == "0-0-1"]
healthy_patients = healthy_patients[~healthy_patients.apply(lambda row: f"pid_{row['pid']}_study_yr_{row['study_yr']}" in df_not_normal["pid_study_yr"].values, axis=1)]
healthy_patients = healthy_patients.sample(n=num_to_replace, random_state=42)

# Funktion zum Löschen der nicht normalen Daten
def delete_not_normal_data(output_dir, df_not_normal):
    deleted_count = 0
    for folder in df_not_normal["pid_study_yr"]:
        folder_path = os.path.join(output_dir, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            logging.info(f"Gelöscht: {folder_path}")
            deleted_count += 1
        else:
            logging.warning(f"Nicht gefunden: {folder_path}")
    return deleted_count

# Funktion zur Auswahl und Speicherung von 3 aufeinanderfolgenden DICOM-Slices in separatem Ordner
def extract_and_save_slices(dicom_folder, pid, study_yr, output_dir):
    try:
        dicom_files = glob(os.path.join(dicom_folder, "*.dcm"))
        if not dicom_files:
            logging.warning(f"Keine DICOM-Dateien gefunden in {dicom_folder}")
            return False

        # DICOM-Dateien laden und nach InstanceNumber sortieren
        dicom_data = [pydicom.dcmread(f) for f in dicom_files]
        dicom_data.sort(key=lambda x: float(x.InstanceNumber))

        # Sicherstellen, dass genügend Slices vorhanden sind
        Z = len(dicom_data)
        if Z < 5:
            logging.warning(f"Zu wenige Slices für PID {pid}, wird übersprungen.")
            return False

        # Start-Slice zufällig aus dem mittleren Bereich mit 5% Varianz wählen
        mid_z = Z // 2
        variance = int(Z * 0.05)  # 5% der Gesamthöhe als Varianz
        start_idx = max(0, min(mid_z - random.randint(0, variance), Z - 3))  # Stelle sicher, dass genug Slices folgen

        # Wähle drei aufeinanderfolgende Slices
        slice_indices = [start_idx, start_idx + 1, start_idx + 2]

        # Zielordner für die Slices erstellen
        patient_output_dir = os.path.join(output_dir, f"pid_{pid}_study_yr_{study_yr}")
        os.makedirs(patient_output_dir, exist_ok=True)

        # Ausgewählte Slices kopieren
        for i, slice_idx in enumerate(slice_indices):
            source_path = dicom_files[slice_idx]
            destination_path = os.path.join(patient_output_dir)
            shutil.copy2(source_path, destination_path)
            logging.info(f"Gespeichert: {destination_path}")

        return True

    except Exception as e:
        logging.error(f"Fehler beim Verarbeiten von PID {pid}: {e}")
        return False

# 1 Entferne nicht normale Daten
deleted_count = delete_not_normal_data(output_path_slices, df_not_normal)
logging.info(f"Insgesamt {deleted_count} nicht normale Patienten entfernt.")

# 2 Ersetze gelöschte Patienten durch neue gesunde
for _, row in tqdm(healthy_patients.iterrows(), total=len(healthy_patients), desc="Ersetze gelöschte Patienten", unit="Patienten"):
    pid = str(row["pid"])
    study_yr = str(row["study_yr"])

    patient_folder = os.path.join(dicom_input_dir, pid)
    if not os.path.exists(patient_folder):
        logging.warning(f"Patientenordner nicht gefunden: {patient_folder}")
        continue

    # Finde den passenden DICOM-Ordner (basierend auf Study Year)
    study_folder = None
    for folder in os.listdir(patient_folder):
        if folder.startswith("01-02-") and folder.split("-")[2] == str(1999 + int(study_yr)):
            study_folder = os.path.join(patient_folder, folder)
            break

    if not study_folder:
        logging.warning(f"Kein passender Study-Year-Ordner gefunden für PID {pid}, Study_Year {study_yr}")
        continue

    # Finde das tiefste Verzeichnis mit DICOM-Dateien
    dicom_folder = study_folder
    while os.path.isdir(dicom_folder):
        subfolders = sorted(os.listdir(dicom_folder))
        if not subfolders:
            break
        dicom_folder = os.path.join(dicom_folder, subfolders[0])
        if glob(os.path.join(dicom_folder, "*.dcm")):
            break

    # Falls keine DICOM-Dateien gefunden wurden, weiter zum nächsten Patienten
    if not glob(os.path.join(dicom_folder, "*.dcm")):
        logging.warning(f"Keine DICOM-Dateien für PID {pid} gefunden.")
        continue

    # Extrahiere und speichere 3 aufeinanderfolgende Slices aus dem DICOM-Volumen in den Zielordner
    extract_and_save_slices(dicom_folder, pid, study_yr, output_path_slices)

logging.info("Fertig! Alle gelöschten Patienten wurden ersetzt.")
