import os
import pandas as pd

# Konfiguration
CSV_PATH = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\nlst_subset_v5.csv"
DATA_PATH = r"D:\thesis_robert\subset_v2_seg"

def find_missing_pids(csv_path, data_path):
    """
    Findet PIDs und Study Years, die in der CSV-Datei sind, aber nicht im Datenverzeichnis existieren.

    Args:
        csv_path (str): Pfad zur CSV-Datei mit den PIDs und Study Years.
        data_path (str): Verzeichnis, das segmentierte Dateien enthält.

    Returns:
        None
    """
    # CSV-Datei einlesen
    df = pd.read_csv(csv_path)
    if not {"pid", "study_yr"}.issubset(df.columns):
        print("Die CSV-Datei muss die Spalten 'pid' und 'study_yr' enthalten.")
        return

    csv_entries = set((str(row["pid"]), str(row["study_yr"])) for _, row in df.iterrows())

    # Alle PIDs und Study Years aus dem Verzeichnis extrahieren
    dir_entries = set()
    for file_name in os.listdir(data_path):
        if file_name.endswith(".nii.gz"):
            parts = file_name.split("_")
            if len(parts) > 3 and parts[0] == "pid":
                pid = parts[1]
                study_yr = parts[4]
                dir_entries.add((pid, study_yr))

    # Fehlende Einträge berechnen
    missing_entries = csv_entries - dir_entries

    # Fehlende Einträge ausgeben
    if missing_entries:
        print("Fehlende PIDs und Study Years:")
        for pid, study_yr in sorted(missing_entries):
            print(f"PID: {pid}, Study Year: {study_yr}")
    else:
        print("Alle PIDs und Study Years aus der CSV sind im Verzeichnis vorhanden.")

if __name__ == "__main__":
    find_missing_pids(CSV_PATH, DATA_PATH)