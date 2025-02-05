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
MODELNAME = "R231"           # "R231" oder "R231CovidWeb"
MODELPATH = r"C:\Users\rbarbir\AppData\Roaming\Python\Python311\site-packages\lungmask\unet_r231-d5d2fc3d.pth"
SAVE_MASKED_VOLUME = True    # True => mask * original
FORCE_CPU = False            # True => CPU, False => GPU

###############################################################################
logging.basicConfig(level=logging.INFO)

def find_matching_file(pid, study_yr, input_dir):
    """
    Findet eine Datei im Verzeichnis, die mit der PID und dem Study Year beginnt
    und mit ".nii.gz" endet.

    Args:
        pid (str): Patient ID
        study_yr (str): Study Year
        input_dir (str): Verzeichnis mit Dateien

    Returns:
        str: Pfad zur gefundenen Datei oder None
    """
    for fname in os.listdir(input_dir):
        if fname.startswith(f"pid_{pid}_study_yr_{study_yr}") and fname.endswith(".nii.gz"):
            return os.path.join(input_dir, fname)
    return None

def is_already_segmented(pid, study_yr, output_dir):
    """
    Prüft, ob die segmentierte Datei bereits im Output-Ordner vorhanden ist.

    Args:
        pid (str): Patient ID
        study_yr (str): Study Year
        output_dir (str): Output-Verzeichnis

    Returns:
        bool: True, wenn die Datei bereits existiert, sonst False
    """
    for fname in os.listdir(output_dir):
        if fname.startswith(f"pid_{pid}_study_yr_{study_yr}") and fname.endswith("_lungmaskedvol.nii.gz"):
            return True
    return False

def main():
    logging.info(f"Starte Lungen-Segmentierung. input_dir={INPUT_DIR}, model={MODELNAME}")

    # 1) CSV einlesen
    if not os.path.exists(CSV_PATH):
        logging.error(f"CSV {CSV_PATH} nicht gefunden. Abbruch.")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    if not {"pid", "study_yr"}.issubset(df.columns):
        logging.error("CSV muss mind. Spalten: pid, study_yr enthalten.")
        sys.exit(1)

    # 2) output_dir anlegen
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3) lungmask Model laden
    logging.info(f"Lade lungmask model: arch='unet', model={MODELNAME}")
    model = mask.get_model(
        modelname=MODELNAME,
        modelpath=MODELPATH
    )

    total_processed = 0
    skipped = 0
    for i, row in df.iterrows():
        pid = str(row["pid"])
        study_yr = str(row["study_yr"])

        # Überspringe, wenn bereits segmentiert
        if is_already_segmented(pid, study_yr, OUTPUT_DIR):
            logging.info(f"Datei für PID={pid}, Study Year={study_yr} bereits segmentiert => Überspringe.")
            skipped += 1
            continue

        # Suche nach passender Datei
        fpath = find_matching_file(pid, study_yr, INPUT_DIR)
        if not fpath:
            logging.warning(f"Keine passende Datei für PID={pid}, Study Year={study_yr} gefunden => Überspringe.")
            continue

        logging.info(f"Segmentiere => {fpath}")

        # 4) Einlesen
        sitk_img = sitk.ReadImage(fpath)

        # 5) lungmask.apply => shape(D,H,W) 0/1
        mask_arr = mask.apply(sitk_img, model, force_cpu=FORCE_CPU)

        if SAVE_MASKED_VOLUME:
            # mask * original
            orig_arr = sitk.GetArrayFromImage(sitk_img)
            masked_arr = orig_arr * mask_arr
            masked_sitk = sitk.GetImageFromArray(masked_arr)
            masked_sitk.CopyInformation(sitk_img)

            out_name = os.path.basename(fpath).replace(".nii", "_lungmaskedvol.nii").replace(".gz", "_lungmaskedvol.nii.gz")
            out_path = os.path.join(OUTPUT_DIR, out_name)
            sitk.WriteImage(masked_sitk, out_path)
            logging.info(f"Maskiertes Volumen gespeichert => {out_path}")
        else:
            # nur die Maske speichern
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