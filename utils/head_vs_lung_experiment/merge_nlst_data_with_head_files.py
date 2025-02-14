import os
import sys
import csv
import shutil
import pandas as pd

CSV_IN = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\nlst_subset_v5.csv"
SOURCE_DIR = r"D:\thesis_robert\NLST_subset_v5_nifti_3mm_Voxel"
DEST_DIR   = r"D:\thesis_robert\test_data_folder\head_vs_lung"

CSV_APPEND = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\head_vs_lung\head_vs_lung_dataset.csv"


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

    df = pd.read_csv(CSV_IN)

    df_lung = df[df['combination'] == "0-0-1"]
    if len(df_lung)==0:
        print("Keine Einträge mit combination=0-0-1 gefunden.")
        sys.exit(0)

    print(f"Gefunden: {len(df_lung)} Einträge mit combination=0-0-1.")

    append_mode = 'a' if os.path.exists(CSV_APPEND) else 'w'
    csvfile = open(CSV_APPEND, mode=append_mode, newline='', encoding='utf-8')
    writer = csv.writer(csvfile)

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

        dest_path = os.path.join(DEST_DIR, filename)
        
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
