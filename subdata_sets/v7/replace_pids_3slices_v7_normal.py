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

# Maximale Anzahl der gewünschten Patienten
TARGET_PATIENT_COUNT = 108

# Nicht normale Patienten laden
df_not_normal = pd.read_excel(not_normal_file, header=None)
df_not_normal.columns = ["filename"]
df_not_normal["pid_study_yr"] = df_not_normal["filename"].str.replace(".nii.gz.nii", "", regex=False)
not_normal_set = set(df_not_normal["pid_study_yr"].tolist())

# Vorhandene Ordner abrufen
existing_folders = set(os.listdir(output_path_slices))
current_patient_count = len(existing_folders)

# **NEU**: Zuerst die Ordner löschen, die in "not_normal" gelistet sind
folders_to_remove = existing_folders.intersection(not_normal_set)
if folders_to_remove:
    logging.info(f"Es werden {len(folders_to_remove)} nicht normale Ordner entfernt...")
    for folder in folders_to_remove:
        folder_path = os.path.join(output_path_slices, folder)
        try:
            shutil.rmtree(folder_path)
            logging.info(f"Ordner entfernt: {folder_path}")
        except Exception as e:
            logging.error(f"Fehler beim Löschen von {folder_path}: {e}")

# Nach dem Löschen erneut den aktuellen Stand abrufen
existing_folders = set(os.listdir(output_path_slices))
current_patient_count = len(existing_folders)
logging.info(f"Aktuelle Anzahl an Patientenordnern nach Löschvorgang: {current_patient_count}")

# Falls nach dem Löschen bereits 108 (oder mehr) vorhanden sind, beenden
if current_patient_count >= TARGET_PATIENT_COUNT:
    logging.info(f"Es sind bereits {current_patient_count} Ordner vorhanden. Kein weiterer Patient wird hinzugefügt.")
    exit(0)

# Patientenliste aus CSV-Datei laden
df = pd.read_csv(csv_file, sep=";")

# Gesunde Patienten auswählen (combination = "0-0-1"), die NICHT in `not_normal` sind und noch nicht im Zielverzeichnis existieren
healthy_patients = df[df["combination"] == "0-0-1"]

healthy_patients = healthy_patients[
    ~healthy_patients.apply(lambda row: f"pid_{row['pid']}_study_yr_{row['study_yr']}" in not_normal_set, axis=1)
]
healthy_patients = healthy_patients[
    ~healthy_patients.apply(lambda row: f"pid_{row['pid']}_study_yr_{row['study_yr']}" in existing_folders, axis=1)
]

# Anzahl der noch benötigten Patienten
missing_patients = TARGET_PATIENT_COUNT - current_patient_count

# Falls zu wenig gesunde Patienten übrig sind, begrenzen
num_available = len(healthy_patients)
num_to_sample = min(num_available, missing_patients)

if num_to_sample < missing_patients:
    logging.warning(f"Nur {num_to_sample} gesunde Patienten verfügbar, aber {missing_patients} benötigt!")

# Zufällig die benötigten Patienten auswählen
healthy_patients = healthy_patients.sample(n=num_to_sample, random_state=42)

# **DICOM-Slices extrahieren und speichern**
def extract_and_save_slices(dicom_folder, pid, study_yr, output_dir):
    try:
        dicom_files = glob(os.path.join(dicom_folder, "*.dcm"))
        if not dicom_files:
            logging.warning(f"Keine DICOM-Dateien gefunden in {dicom_folder}")
            return False

        dicom_data = [pydicom.dcmread(f) for f in dicom_files]
        dicom_data.sort(key=lambda x: float(x.InstanceNumber))

        Z = len(dicom_data)
        if Z < 5:
            logging.warning(f"Zu wenige Slices für PID {pid}, wird übersprungen.")
            return False

        mid_z = Z // 2
        variance = int(Z * 0.05)
        start_idx = max(0, min(mid_z - random.randint(0, variance), Z - 3))

        slice_indices = [start_idx, start_idx + 1, start_idx + 2]

        patient_output_dir = os.path.join(output_dir, f"pid_{pid}_study_yr_{study_yr}")
        os.makedirs(patient_output_dir, exist_ok=True)

        for i, slice_idx in enumerate(slice_indices):
            source_path = dicom_files[slice_idx]
            destination_path = os.path.join(patient_output_dir)
            shutil.copy2(source_path, destination_path)
            logging.info(f"Gespeichert: {os.path.join(destination_path, os.path.basename(source_path))}")

        return True

    except Exception as e:
        logging.error(f"Fehler beim Verarbeiten von PID {pid}: {e}")
        return False

# **Neue Patienten hinzufügen, bis 108 erreicht sind**
added_count = 0
for _, row in tqdm(healthy_patients.iterrows(), total=num_to_sample, desc="Füge neue Patienten hinzu", unit="Patienten"):
    if current_patient_count + added_count >= TARGET_PATIENT_COUNT:
        break  # Stoppe, wenn 108 erreicht sind

    pid = str(row["pid"])
    study_yr = str(row["study_yr"])

    patient_folder = os.path.join(dicom_input_dir, pid)
    if not os.path.exists(patient_folder):
        logging.warning(f"Patientenordner nicht gefunden: {patient_folder}")
        continue

    study_folder = None
    for folder in os.listdir(patient_folder):
        # Bsp: "01-02-2000"
        # Prüfe, ob das Format "01-02-{YYYY}" ist und das Jahr stimmt
        if folder.startswith("01-02-") and folder.split("-")[2] == str(1999 + int(study_yr)):
            study_folder = os.path.join(patient_folder, folder)
            break

    if not study_folder:
        logging.warning(f"Kein passender Study-Year-Ordner gefunden für PID {pid}, Study_Year {study_yr}")
        continue

    dicom_folder = study_folder
    while os.path.isdir(dicom_folder):
        subfolders = sorted(os.listdir(dicom_folder))
        if not subfolders:
            break
        dicom_folder = os.path.join(dicom_folder, subfolders[0])
        # Sobald wir .dcm-Dateien finden, brechen wir ab
        if glob(os.path.join(dicom_folder, "*.dcm")):
            break

    # Prüfe nochmal, ob es nun .dcm Dateien gibt
    if not glob(os.path.join(dicom_folder, "*.dcm")):
        logging.warning(f"Keine DICOM-Dateien für PID {pid} gefunden.")
        continue

    if extract_and_save_slices(dicom_folder, pid, study_yr, output_path_slices):
        added_count += 1

# **Finale Anzahl überprüfen**
final_count = len(os.listdir(output_path_slices))
logging.info(f"Finale Anzahl an Patientenordnern: {final_count}/{TARGET_PATIENT_COUNT}")

if final_count < TARGET_PATIENT_COUNT:
    logging.warning(f"Es fehlen noch {TARGET_PATIENT_COUNT - final_count} Patienten!")

logging.info("Fertig! Alle nicht normalen Patienten wurden entfernt und die Gesamtanzahl wurde geprüft/aufgefüllt.")
