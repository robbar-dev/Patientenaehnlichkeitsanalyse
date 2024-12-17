import os
from tqdm import tqdm
from SimpleITK import ImageSeriesReader, WriteImage
import pydicom
import json
from concurrent.futures import ThreadPoolExecutor

# Pfade
json_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V2\nlst_subset_v2.json"
output_dir = r"D:\thesis_robert\subset_v2\NLST_subset_v2_nifti_unverarbeitet"

# Zielverzeichnis erstellen, falls es nicht existiert
os.makedirs(output_dir, exist_ok=True)

# Lade die JSON-Daten
with open(json_path, "r") as f:
    data = json.load(f)

# Initialisierung
successful = []
failed = []

# Funktion zur Konvertierung
def process_entry(entry):
    pid = entry.get("pid", "UNBEKANNT")
    study_yr = entry.get("study_yr", "UNBEKANNT")
    dicom_path = entry.get("dicom_path", None)

    if not dicom_path:
        return f"WARNUNG: Kein DICOM-Pfad für PID {pid}, Study Year {study_yr}."

    # Zielpfad und Dateiname erstellen
    file_name_out = f"pid_{pid}_study_yr_{study_yr}.nii.gz"
    file_path_out = os.path.join(output_dir, file_name_out)

    # Überspringen, wenn Datei bereits existiert
    if os.path.exists(file_path_out):
        return f"ÜBERSPRUNGEN: {file_name_out} existiert bereits."

    try:
        dicom_files = os.listdir(dicom_path)
        if not dicom_files:
            return f"FEHLER: Keine Dateien im Ordner {dicom_path}."

        test_dicom_file = os.path.join(dicom_path, dicom_files[0])
        ds = pydicom.dcmread(test_dicom_file)

        reader = ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
        if not dicom_names:
            return f"FEHLER: Keine DICOM-Serie erkannt in {dicom_path}."

        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        WriteImage(image, file_path_out)
        successful.append((pid, study_yr))
        return f"ERFOLG: {file_name_out} geschrieben."
    except Exception as e:
        failed.append((pid, study_yr))
        return f"FEHLER: Konvertierung fehlgeschlagen für {dicom_path}: {e}"

# Parallelisierte Verarbeitung
with ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_entry, data), total=len(data), desc="Konvertiere DICOM zu NIfTI"))

# Ergebnisse ausgeben
for result in results:
    print(result)

# Zusammenfassung
print(f"Konvertiert: {len(successful)} | Fehlgeschlagen: {len(failed)} | Gesamt: {len(successful) + len(failed)}")

# Log-Datei schreiben
log_file_path = os.path.join(output_dir, "conversion_log.txt")
with open(log_file_path, "w") as log_file:
    if failed:
        log_file.write("Fehlgeschlagene Konvertierungen:\n")
        for pid, study_yr in failed:
            log_file.write(f"PID: {pid}, Study Year: {study_yr}\n")
    else:
        log_file.write("Alle DICOM-Dateien wurden erfolgreich konvertiert.\n")

print(f"Log-Datei erstellt unter: {log_file_path}")
