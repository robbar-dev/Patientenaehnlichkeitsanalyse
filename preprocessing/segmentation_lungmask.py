#!/usr/bin/env python3
"""
simple_segmentation_lungmask.py
-------------------------------
Ein fest konfiguriertes Skript für die Lungen­segmentierung via lungmask (pretrained),
OHNE argparse, stattdessen sind Pfade etc. fest im Code.

Zudem wird eine CSV eingelesen, in der pro Zeile [pid, study_yr, ...] steht,
und wir bauen daraus den Dateinamen:
   pid_{pid}_study_yr_{study_yr}.nii.gz
Wenn die Datei existiert, segmentieren wir sie und speichern das Ergebnis
(_lungmaskedvol.nii.gz) ins output_dir.

Achtung:
 - Du musst ggf. den Dateinamen anpassen, falls Dein Schema anders ist
   (z.B. pid_{pid}_study_yr_{study_yr}.nii.gz).
 - Du kannst optional `SAVE_MASKED_VOLUME=True/False` toggeln.
"""

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
INPUT_DIR = r"D:\thesis_robert\subset_v2_trimmed"
OUTPUT_DIR = r"D:\thesis_robert\subset_v2_seg"

CSV_PATH   = r"D:\thesis_robert\my_dataset.csv"   # Enthält Spalten [pid, study_yr, ...] 

# 2) lungmask-Parameter
MODELNAME = "R231"           # "R231" oder "R231CovidWeb"
MODELPATH = r"C:\Users\rbarbir\AppData\Roaming\Python\Python311\site-packages\lungmask\unet_r231-d5d2fc3d.pth"
SAVE_MASKED_VOLUME = True    # True => mask * original
FORCE_CPU = False            # True => CPU, False => GPU

###############################################################################
logging.basicConfig(level=logging.INFO)

def main():
    logging.info(f"Starte Lungen-Segmentierung. input_dir={INPUT_DIR}, model={MODELNAME}")

    # 1) CSV einlesen
    if not os.path.exists(CSV_PATH):
        logging.error(f"CSV {CSV_PATH} nicht gefunden. Abbruch.")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    if not {"pid","study_yr"}.issubset(df.columns):
        logging.error("CSV muss mind. Spalten: pid, study_yr enthalten.")
        sys.exit(1)

    # 2) output_dir anlegen
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3) lungmask Model laden
    logging.info(f"lade lungmask model: arch='unet', model={MODELNAME}")
    model = mask.get_model(
        modelname=MODELNAME,
        modelpath=MODELPATH
    )

    total_processed = 0
    for i, row in df.iterrows():
        pid = str(row["pid"])
        study_yr = str(row["study_yr"])

        # Baue Dateiname => z.B. pid_XXXX_study_yr_YYYY.nii.gz
        # Passen an Dein Schema an
        fname = f"pid_{pid}_study_yr_{study_yr}.nii.gz"
        fpath = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(fpath):
            logging.warning(f"Datei {fpath} existiert nicht => Überspringe.")
            continue

        logging.info(f"Segmentiere => {fpath}")

        # 4) Einlesen
        sitk_img = sitk.ReadImage(fpath)

        # 5) lungmask.apply => shape(D,H,W) 0/1
        mask_arr = mask.apply(sitk_img, model, force_cpu=FORCE_CPU)
        # Optional: np.unique(mask_arr) inspizieren => 0/1 oder 0/1/2/3

        if SAVE_MASKED_VOLUME:
            # mask * original
            orig_arr = sitk.GetArrayFromImage(sitk_img)
            masked_arr = orig_arr * mask_arr
            masked_sitk = sitk.GetImageFromArray(masked_arr)
            masked_sitk.CopyInformation(sitk_img)

            out_name = fname.replace(".nii", "_lungmaskedvol.nii").replace(".gz", "_lungmaskedvol.nii.gz")
            out_path = os.path.join(OUTPUT_DIR, out_name)
            sitk.WriteImage(masked_sitk, out_path)
            logging.info(f"Maskiertes Volumen gespeichert => {out_path}")
        else:
            # nur die Maske speichern
            mask_sitk = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
            mask_sitk.CopyInformation(sitk_img)
            out_name = fname.replace(".nii", "_lungmask.nii").replace(".gz", "_lungmask.nii.gz")
            out_path = os.path.join(OUTPUT_DIR, out_name)
            sitk.WriteImage(mask_sitk, out_path)
            logging.info(f"Maske gespeichert => {out_path}")

        total_processed += 1

    logging.info(f"FERTIG. Insgesamt {total_processed} Dateien segmentiert.")


if __name__ == "__main__":
    main()

# python3.11 preprocessing\simple_segmentation_lungmask.py