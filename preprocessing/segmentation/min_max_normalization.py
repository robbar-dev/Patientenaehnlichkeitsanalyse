"""
Min-Max Normalisierung für alle .nii/.nii.gz in einem Ordner.
Speichert das normalisierte Volume in OUTPUT_DIR, wobei Intensitäten => [0..1].

"""

import os
import sys
import argparse
import logging
import numpy as np
import nibabel as nib

def get_parser():
    parser = argparse.ArgumentParser(description="Min–Max Normalisierung aller NIfTI-Dateien in input_dir => output_dir.")
    parser.add_argument("--input_dir", required=True, help="Pfad zum Verzeichnis mit den Eingabe-.nii / .nii.gz")
    parser.add_argument("--output_dir", required=True, help="Ausgabeverzeichnis für die normalisierten NIfTI-Dateien")
    parser.add_argument("--epsilon", type=float, default=1e-7, help="kleiner Wert, um Division durch 0 zu vermeiden")
    return parser

def minmax_normalize_volume(volume_np, eps=1e-7):
    """
    volume_np: np.ndarray (H,W,D) oder (C,H,W,D)
    => returns volume_np skaliert in [0..1]
    """
    vmin = volume_np.min()
    vmax = volume_np.max()
    rng = vmax - vmin
    if rng < eps:
        # Falls fast konstantes Volumen => setze alles auf 0
        return np.zeros_like(volume_np)
    else:
        return (volume_np - vmin)/(rng)

def main():
    parser = get_parser()
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    eps = args.epsilon

    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    # Liste aller NIfTI-Dateien
    files = [f for f in os.listdir(input_dir) if (f.endswith(".nii") or f.endswith(".nii.gz"))]
    if not files:
        logging.warning(f"Keine .nii/.nii.gz Dateien in {input_dir}")
        sys.exit(0)

    processed = 0
    for i, fname in enumerate(files):
        in_path = os.path.join(input_dir, fname)
        logging.info(f"[{i+1}/{len(files)}] Lade NIfTI: {in_path}")

        try:
            img = nib.load(in_path)
            volume_np = img.get_fdata(dtype=np.float32)  # => float32
            norm_vol = minmax_normalize_volume(volume_np, eps)

            # gleiche Affine + Header übernehmen
            norm_img = nib.Nifti1Image(norm_vol, affine=img.affine, header=img.header)
            # Dateiname anpassen
            out_name = fname.replace(".nii","_minmax.nii").replace(".gz","_minmax.nii.gz")
            out_path = os.path.join(output_dir, out_name)

            nib.save(norm_img, out_path)
            logging.info(f"Gespeichert => {out_path}")
            processed += 1

        except Exception as e:
            logging.error(f"Fehler bei Datei {fname}: {e}")
            continue

    logging.info(f"Fertig. Insgesamt {processed}/{len(files)} Volumes normalisiert.")

if __name__ == "__main__":
    main()

# python3.11 preprocessing\segmentation\min_max_normalization.py --input_dir "D:\\thesis_robert\\NLST_subset_v5_SEG_nifti_1_5mm_Voxel" --output_dir "D:\\thesis_robert\\NLST_subset_v5_SEG_NORM_nifti_1_5mm_Voxel"