#!/usr/bin/env python3

"""
rotate_nifti.py
---------------
Dieses Skript liest alle .nii / .nii.gz Dateien aus input_dir, macht eine
Rotation in der x-y-Ebene (z. B. 90° oder 180°) und speichert das Ergebnis
im output_dir.

Dadurch siehst Du sicher, dass sich die Bilder ändern.
Du kannst dann z. B. den Output für Segmentierung, etc. verwenden.

Beispielaufruf:
  python rotate_nifti.py --input_dir "Pfad" --output_dir "Pfad_out" --k 1
"""

import os
import sys
import argparse
import logging

import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO)

def get_parser():
    parser = argparse.ArgumentParser(
        description="Rotiert NIfTI-Bilder in-plane via np.rot90 (k=1..3)."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Eingabeordner mit .nii / .nii.gz-Dateien.")
    parser.add_argument("--output_dir", required=True,
                        help="Ausgabeordner für die rotierten NIfTI-Dateien.")
    parser.add_argument("--k", type=int, default=1,
                        help="Anzahl 90°-Schritte (1=90°, 2=180°, 3=270°). Default=1.")
    return parser

def rotate_inplane_npy(arr, k=1):
    """
    arr: Numpy-Array shape (D, H, W) => wir rotieren in-plane (H,W).
    k=1 => 90°, k=2 => 180°, k=3 => 270° im Uhrzeigersinn (Standard np.rot90).
    Rückgabe: rotiertes Array (D, H, W)
    """
    # np.rot90(..., axes=(1,2)) rotiert in der (H,W)-Ebene
    # default np.rot90 ist gegen den Uhrzeigersinn => 
    # falls Du lieber im Uhrzeigersinn willst, kannst Du k=3 statt k=1 usw.
    rotated = np.rot90(arr, k=k, axes=(1, 2))
    return rotated

def main():
    parser = get_parser()
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    k = args.k

    logging.info(f"Starte Rotation in-plane: input_dir={input_dir}, k={k}")
    os.makedirs(output_dir, exist_ok=True)

    # 1) Liste NIfTI
    files = sorted([
        f for f in os.listdir(input_dir)
        if (f.endswith(".nii") or f.endswith(".nii.gz"))
    ])
    if not files:
        logging.warning("Keine NIfTI-Dateien gefunden.")
        sys.exit(0)

    total = 0
    for idx, fname in enumerate(files):
        in_path = os.path.join(input_dir, fname)
        logging.info(f"[{idx+1}/{len(files)}] Lade {in_path}")

        # 2) SITK => Numpy
        sitk_img = sitk.ReadImage(in_path)
        arr = sitk.GetArrayFromImage(sitk_img)  # shape (D,H,W)

        # 3) Rotation
        arr_rot = rotate_inplane_npy(arr, k=k)
        logging.info(f"Original shape={arr.shape}, rotated shape={arr_rot.shape}")

        # 4) Zurück in SITK
        out_img = sitk.GetImageFromArray(arr_rot)
        # Spacing etc. übernehmen wir (bis auf die in-plane-Orientierung),
        # aber pass auf, dass du direction/origin übernimmst => teils kann
        # das verwirrend sein, da die Achsen jetzt getauscht sind.
        # => wir übernehmen einfach die Info, damit Z,Spacing etc. gleichbleibt.
        out_img.CopyInformation(sitk_img)

        # 5) Speichern
        out_name = fname.replace(".nii", f"_rot{k}.nii").replace(".gz", f"_rot{k}.nii.gz")
        out_path = os.path.join(output_dir, out_name)
        sitk.WriteImage(out_img, out_path)
        logging.info(f"Gespeichert: {out_path}")

        total += 1

    logging.info(f"Fertig. {total} Dateien rotiert in-plane (k={k}).")

if __name__=="__main__":
    main()


# python3.11 preprocessing\nifti_orientation.py --input_dir "D:\thesis_robert\Segmentation_Test\V2_unverarbeitet_Data\testdata_v2_unverarbeitet" --output_dir "D:\thesis_robert\Segmentation_Test\V2_unverarbeitet_Data\testdata_v2_unverarbeitet_orient" --k 2
