import os
import sys
import logging
import pandas as pd
import SimpleITK as sitk
import numpy as np

# lungmask import
from lungmask import mask

###############################################################################
# Konfiguration
###############################################################################
# 1) Pfade
INPUT_DIR = r"D:\thesis_robert\NLST_subset_v7_anomalie_resampled_orientation"
OUTPUT_DIR = r"D:\thesis_robert\NLST_subset_v7_anomalie_resampled_SEG"

CSV_PATH = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V7\nlst_subset_v7_anomalie_data.csv"

# 2) lungmask-Parameter
MODELNAME = "R231"  # "R231" oder "R231CovidWeb"
MODELPATH = r"C:\Users\rbarbir\AppData\Roaming\Python\Python311\site-packages\lungmask\unet_r231-d5d2fc3d.pth"
SAVE_MASKED_VOLUME = True  # True => mask * original
FORCE_CPU = False  # True => CPU, False => GPU

###############################################################################
logging.basicConfig(level=logging.INFO)

def find_matching_file(pid, study_yr, input_dir):
    for fname in os.listdir(input_dir):
        if fname.startswith(f"pid_{pid}_study_yr_{study_yr}") and fname.endswith(".nii.gz"):
            return os.path.join(input_dir, fname)
    return None

def is_already_segmented(pid, study_yr, output_dir):
    for fname in os.listdir(output_dir):
        if fname.startswith(f"pid_{pid}_study_yr_{study_yr}") and fname.endswith("_lungmaskedvol.nii.gz"):
            return True
    return False

def main():
    logging.info(f"Starte Lungen-Segmentierung. input_dir={INPUT_DIR}, model={MODELNAME}")

    if not os.path.exists(CSV_PATH):
        logging.error(f"CSV {CSV_PATH} nicht gefunden. Abbruch.")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    if not {"pid", "study_yr"}.issubset(df.columns):
        logging.error("CSV muss mind. Spalten: pid, study_yr enthalten.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info(f"Lade lungmask model: arch='unet', model={MODELNAME}")
    model = mask.get_model(modelname=MODELNAME, modelpath=MODELPATH)

    total_processed = 0
    skipped = 0

    for _, row in df.iterrows():
        pid = str(row["pid"])
        study_yr = str(row["study_yr"])

        if is_already_segmented(pid, study_yr, OUTPUT_DIR):
            logging.info(f"Datei für PID={pid}, Study Year={study_yr} bereits segmentiert => Überspringe.")
            skipped += 1
            continue

        fpath = find_matching_file(pid, study_yr, INPUT_DIR)
        if not fpath:
            logging.warning(f"Keine passende Datei für PID={pid}, Study Year={study_yr} gefunden => Überspringe.")
            continue

        logging.info(f"Segmentiere => {fpath}")

        # 1) Einlesen
        sitk_img = sitk.ReadImage(fpath)
        orig_arr = sitk.GetArrayFromImage(sitk_img)

        # 2) Debugging: Prüfe, ob das Originalbild gültig ist
        logging.info(f"Originalbild {fpath}: Min={orig_arr.min()}, Max={orig_arr.max()}, Mean={orig_arr.mean()}")
        if np.max(orig_arr) == 0:
            logging.error(f"ACHTUNG: Original NIfTI {fpath} ist komplett schwarz! Überprüfe das Input-File.")
            continue

        # 3) lungmask anwenden
        mask_arr = mask.apply(sitk_img, model, force_cpu=FORCE_CPU)

        # 4) Debugging: Prüfe, ob die Maske gültig ist
        logging.info(f"Masken-Statistik {fpath}: Min={mask_arr.min()}, Max={mask_arr.max()}, Mean={mask_arr.mean()}")
        if np.max(mask_arr) == 0:
            logging.error(f"ACHTUNG: Die generierte Maske für {fpath} ist komplett schwarz! Möglicherweise erkennt lungmask keine Lunge.")
            continue

        # 5) Maske mit Originalbild kombinieren
        if SAVE_MASKED_VOLUME:
            masked_arr = np.where(mask_arr > 0, orig_arr, 0)  # Alternative zu orig_arr * mask_arr
            masked_sitk = sitk.GetImageFromArray(masked_arr)
            masked_sitk.CopyInformation(sitk_img)

            out_name = os.path.basename(fpath).replace(".nii", "_lungmaskedvol.nii").replace(".gz", "_lungmaskedvol.nii.gz")
            out_path = os.path.join(OUTPUT_DIR, out_name)
            sitk.WriteImage(masked_sitk, out_path)
            logging.info(f"Maskiertes Volumen gespeichert => {out_path}")
        else:
            mask_sitk = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
            mask_sitk.CopyInformation(sitk_img)
            out_name = os.path.basename(fpath).replace(".nii", "_lungmask.nii").replace(".gz", "_lungmask.nii.gz")
            out_path = os.path.join(OUTPUT_DIR, out_name)
            sitk.WriteImage(mask_sitk, out_path)
            logging.info(f"Maske gespeichert => {out_path}")

        total_processed += 1

    logging.info(f"FERTIG. Insgesamt {total_processed} Dateien segmentiert. {skipped} Dateien wurden übersprungen.")

if __name__ == "__main__":
    main()


# python3.11 preprocessing\segmentation\segmentation_lungmask.py