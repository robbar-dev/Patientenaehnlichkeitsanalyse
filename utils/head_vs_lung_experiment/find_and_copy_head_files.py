import os
import sys
import csv
import shutil

SOURCE_DIR = r"D:\thesis_david\training_data\tcia_ml\Head-Neck-PET-CT\ct"
DEST_DIR   = r"D:\thesis_robert\test_data_folder\head_vs_lung"
CSV_NAME   = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\head_vs_lung\head_vs_lung_dataset.csv"

MAX_PATIENTS = 254

def parse_patient_folder(foldername):
    """
    Beispiel: "HN-CHUM-001" -> pid="HNCHUM", study="001"
    Wir entfernen den mittleren Bindestrich und extrahieren den Teil nach dem letzten Bindestrich als study_yr.
    """
    parts = foldername.split('-')  # ["HN","CHUM","001"] -> verschiedene Bezeichnungen in HEAD-NECK-DATASET
    if len(parts) < 3:
        return None, None
    
    # pid_part: "HN"+"CHUM" => "HNCHUM"
    pid_part = parts[0] + parts[1]  # z.B. "HNCHUM"
    study_yr = parts[2]            # z.B. "001"
    return pid_part, study_yr

def find_nii_file_in_subdirs(folder):
    """
    Sucht rekursiv in 'folder' nach einer Datei, die auf .nii oder .nii.gz endet,
    und NICHT in einem "TomoTherapy" Ordner liegt.
    Gibt den absoluten Pfad zum ersten Fund zurück oder None.
    """
    for root, dirs, files in os.walk(folder):
        # Wichtig ist, dass "TomoTherapy"- Ordnern geskippt werden 
        if "TomoTherapy" in root:
            continue

        for f in files:
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                return os.path.join(root, f)

    return None

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Quelle {SOURCE_DIR} nicht gefunden.")
        sys.exit(1)

    os.makedirs(DEST_DIR, exist_ok=True)

    csv_path = os.path.join(DEST_DIR, CSV_NAME)
    csv_file = open(csv_path, mode="w", newline='', encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["pid", "study_yr", "combination"])  

    patient_folders = sorted([
        d for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ])

    total_copied = 0

    for folder in patient_folders:
        if total_copied >= MAX_PATIENTS:
            print(f"Limit von {MAX_PATIENTS} Patienten erreicht.")
            break

        folderpath = os.path.join(SOURCE_DIR, folder)
        pid, study = parse_patient_folder(folder)
        if pid is None or study is None:
            print(f"Überspringe Ordner: {folder}")
            continue

        # Finde .nii / .nii.gz aber ignoriere "TomoTherapy"
        nii_path = find_nii_file_in_subdirs(folderpath)
        if nii_path is None:
            print(f"Keine NIfTI-Datei gefunden in: {folderpath} => überspringe")
            continue

        # Zieldateiname bilden -> pid_HNCHUM_study_yr_001.nii.gz
        ext = ".nii.gz" if nii_path.endswith(".nii.gz") else ".nii"
        new_name = f"pid_{pid}_study_yr_{study}{ext}"
        dest_path = os.path.join(DEST_DIR, new_name)

        shutil.copy2(nii_path, dest_path)
        print(f"Kopiert: {nii_path} => {dest_path}")

        writer.writerow([pid, study, "0-0-2"])
        total_copied += 1

    csv_file.close()
    print(f"Fertig. Insgesamt {total_copied} Dateien kopiert.")
    print(f"CSV geschrieben nach {csv_path}")

if __name__ == "__main__":
    main()
