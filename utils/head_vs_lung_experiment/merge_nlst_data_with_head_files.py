#!/usr/bin/env python3
"""
copy_lung_data.py
-----------------
Kopiert alle Lungen-Daten (combination=0-0-1) aus dem existierenden NLST-Datensatz
nach "D:\\thesis_robert\\test_data_folder\\head_vs_lung"
und hängt die entsprechenden Zeilen an
"C:\\Users\\rbarbir\\OneDrive - Brainlab AG\\Dipl_Arbeit\\Datensätze\\head_vs_lung\\head_vs_lung_dataset.csv".

Voraussetzungen:
 - Die Quelldateien liegen im Ordner "D:\\thesis_robert\\NLST_subset_v5_nifti_3mm_Voxel"
   und heißen typischerweise "pid_{pid}_study_yr_{study_yr}.nii.gz".
 - Die CSV "nlst_subset_v5.csv" enthält Spalten [pid, study_yr, combination].
 - Wir filtern combination == "0-0-1".

Ergebnis:
 - Kopierte NIfTI-Dateien im Zielordner
 - Angehängte Zeilen in head_vs_lung_dataset.csv
"""

import os
import sys
import csv
import shutil
import pandas as pd

# 1) Pfade anpassen
CSV_IN = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\nlst_subset_v5.csv"
SOURCE_DIR = r"D:\thesis_robert\NLST_subset_v5_nifti_3mm_Voxel"
DEST_DIR   = r"D:\thesis_robert\test_data_folder\head_vs_lung"

CSV_APPEND = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\head_vs_lung\head_vs_lung_dataset.csv"

# Falls Du lieber "combination=0-0-1" beibehalten willst, nimm VARIANTE 1
# Falls Du Head=0, Lung=1 => nimm VARIANTE 2

USE_ORIGINAL_COMBINATION = True  # VARIANTE 1: "0-0-1"
# USE_ORIGINAL_COMBINATION = False  # VARIANTE 2: "1"

def main():
    if not os.path.exists(CSV_IN):
        print(f"CSV {CSV_IN} nicht gefunden.")
        sys.exit(1)
    if not os.path.exists(SOURCE_DIR):
        print(f"Quelldatenordner {SOURCE_DIR} nicht gefunden.")
        sys.exit(1)

    os.makedirs(DEST_DIR, exist_ok=True)

    # 2) Einlesen CSV
    df = pd.read_csv(CSV_IN)
    # Filtern
    df_lung = df[df['combination'] == "0-0-1"]
    if len(df_lung)==0:
        print("Keine Einträge mit combination=0-0-1 gefunden.")
        sys.exit(0)

    print(f"Gefunden: {len(df_lung)} Einträge mit combination=0-0-1.")

    # 3) CSV_APPEND prüfen
    # => wir müssen an diese CSV Zeilen anhängen
    append_mode = 'a' if os.path.exists(CSV_APPEND) else 'w'
    csvfile = open(CSV_APPEND, mode=append_mode, newline='', encoding='utf-8')
    writer = csv.writer(csvfile)

    # Falls CSV_APPEND noch nicht existierte, Header schreiben
    if append_mode=='w':
        writer.writerow(["pid","study_yr","combination"])

    total_copied = 0

    for i, row in df_lung.iterrows():
        pid = str(row['pid'])
        study_yr = str(row['study_yr'])
        combo_str = row['combination']  # = "0-0-1"

        filename = f"pid_{pid}_study_yr_{study_yr}.nii.gz.nii.gz"
        source_path = os.path.join(SOURCE_DIR, filename)

        if not os.path.exists(source_path):
            print(f"[WARN] Datei {source_path} existiert nicht -> überspringe")
            continue

        # Zielpfad
        dest_path = os.path.join(DEST_DIR, filename)

        # Kopieren
        shutil.copy2(source_path, dest_path)
        print(f"Kopiert: {source_path} => {dest_path}")

        # Head=0, Lung=1 => => "1"
        writer.writerow([pid, study_yr, "0-0-1"])

        total_copied += 1

    csvfile.close()
    print(f"Fertig. Insgesamt {total_copied} Dateien kopiert.")
    print(f"CSV-Einträge appended an {CSV_APPEND}")

if __name__=="__main__":
    main()
