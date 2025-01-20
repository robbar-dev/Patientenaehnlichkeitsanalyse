#!/usr/bin/env python3

"""
segmentation.py
---------------
Beispielskript für eine automatische Lungen­segmentierung mithilfe 'lungmask'.

Requirements:
  pip install lungmask SimpleITK

Ergebnis:
 - Für jedes .nii/.nii.gz im input_dir:
   1) Erzeugt maske => *_lungmask.nii.gz
   2) Optional das maskierte Volumen => *_lungmaskedvol.nii.gz
"""

import os
import sys
import argparse
import logging

import SimpleITK as sitk
import numpy as np

# Wichtig: 'mask' statt 'lungmask'
from lungmask import mask

logging.basicConfig(level=logging.INFO)

def get_parser():
    parser = argparse.ArgumentParser(
        description="Automatische Lungen­segmentierung via lungmask (pretrained)."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Pfad zum Verzeichnis mit .nii / .nii.gz Dateien (CT-Volumes).")
    parser.add_argument("--output_dir", required=True,
                        help="Pfad zum Verzeichnis, in dem die Ergebnisse gespeichert werden.")
    parser.add_argument("--modelname", type=str, default="R231",
                        help="Welches pretrained Model in lungmask verwendet wird (z.B. 'R231', 'R231CovidWeb').")
    parser.add_argument("--save_masked_volume", action='store_true',
                        help="Wenn gesetzt, wird zusätzlich (Maske * OriginalCT) abgespeichert.")
    parser.add_argument("--force_cpu", action='store_true',
                        help="Wenn gesetzt, wird Lungenmaske trotz GPU auf CPU ausgeführt.")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    modelname = args.modelname
    save_masked = args.save_masked_volume
    force_cpu = args.force_cpu

    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Starte Lungen-Segmentierung: input_dir={input_dir}, model={modelname}")

    # 1) Liste aller CT-Dateien
    ct_files = sorted([
        f for f in os.listdir(input_dir)
        if (f.endswith(".nii") or f.endswith(".nii.gz"))
    ])
    if not ct_files:
        logging.warning(f"Keine NIfTI-Dateien in {input_dir} gefunden.")
        sys.exit(0)

    # 2) Lade das pretrained Model via `mask.get_model`
    logging.info(f"Lade pretrained lungmask Model: arch='unet', model='{modelname}'")
    model = mask.get_model(
        modelname="R231", 
        modelpath=r"C:\Users\rbarbir\AppData\Roaming\Python\Python311\site-packages\lungmask\unet_r231-d5d2fc3d.pth"
    )

    # 3) Iteriere über CT-Dateien
    for idx, ct_filename in enumerate(ct_files):
        ct_path = os.path.join(input_dir, ct_filename)
        logging.info(f"[{idx+1}/{len(ct_files)}] Lade CT-Volume: {ct_path}")

        # SimpleITK
        image_sitk = sitk.ReadImage(ct_path)

        # 4) lungmask.apply => 3D-Numpy-Array [D,H,W] mit {0,1}
        logging.info(f"Wende lungmask.apply(...) an, model={modelname}, force_cpu={force_cpu}")
        mask_arr = mask.apply(image_sitk, model, force_cpu=force_cpu)  
        # shape: (D,H,W), 0=Hintergrund, 1=Lunge

        # 5) Maske * Original => maskiertes Volume
        logging.info("Erstelle maskiertes Volume (Maske * Original).")
        original_arr = sitk.GetArrayFromImage(image_sitk)  # (D,H,W)
        masked_arr = original_arr * mask_arr  # => (D,H,W), nur Lunge

        masked_sitk = sitk.GetImageFromArray(masked_arr)
        masked_sitk.CopyInformation(image_sitk)

        # Maskiertes Volumen speichern
        masked_outname = ct_filename.replace(".nii", "_lungmaskedvol.nii").replace(".gz", "_lungmaskedvol.nii.gz")
        masked_path = os.path.join(output_dir, masked_outname)
        sitk.WriteImage(masked_sitk, masked_path)
        logging.info(f"Maskiertes Volumen gespeichert => {masked_path}")

    logging.info("FERTIG. Alle CTs wurden segmentiert.")


if __name__ == "__main__":
    main()

# python3.11 preprocessing\segmentation.py --input_dir "D:\thesis_robert\NLST_subset_v5_nifti_3mm_Voxel" --output_dir "D:\thesis_robert\NLST_subset_v5_nifti_3mm_Voxel_segmented" --modelname "R231" --save_masked_volume
