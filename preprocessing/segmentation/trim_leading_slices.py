import os
import sys
import argparse
import logging

import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO)

"""
Dieses Skript liest alle .nii/.nii.gz-Dateien aus input_dir, ermittelt 
den ersten Slice, in dem genug "Lungen-HU" (z.B. < -300 HU) vorkommt, 
und trimmt alle Slices davor weg. Danach speichert es das reduzierte 
Volume im output_dir. 

-> Serien starten mit Lungenvolumen
-> lungmask kann diese Serien besser segmentieren

Heuristik:
 - HU_SCHWELLE = -300
 - fraction_threshold = 0.05  (5%)
 - Ab der ersten Slice: 
   fraction = #Pixel(HU < -300) / (H*W)
   Wenn fraction > 0.05 => slice_idx= i
   => arr[i:] wird beibehalten
"""

def get_parser():
    parser = argparse.ArgumentParser(
        description="Entfernt führende Slices ohne nennenswerten Lungen-HU-Anteil."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Verzeichnis mit .nii oder .nii.gz Dateien.")
    parser.add_argument("--output_dir", required=True,
                        help="Zielverzeichnis.")
    parser.add_argument("--hu_threshold", type=float, default=-300.0,
                        help="HU-Schwelle für 'Lungen-Pixel'. Default=-300.")
    parser.add_argument("--fraction_threshold", type=float, default=0.05,
                        help="Minimaler Anteil an HU < hu_threshold, damit wir 'Lunge' detektieren. Default=0.05 (5%).")
    return parser

def find_first_lung_slice(arr_3d, hu_thresh=-300, fraction_thresh=0.05):
    """
    arr_3d: (D,H,W)  -> CT in HU
    hu_thresh: z. B. -300
    fraction_thresh: z. B. 0.05 => 5%

    Durchsucht slices ab i=0 => berechnet fraction:
      fraction = sum(arr_3d[i,:,:]< hu_thresh) / (H*W)
    Wenn fraction> fraction_thresh => return i
    return None, wenn keine gefunden
    """
    D, H, W = arr_3d.shape
    for i in range(D):
        slice_arr = arr_3d[i]
        fraction = np.mean(slice_arr < hu_thresh)
        if fraction > fraction_thresh:
            return i
    return None

def main():
    parser = get_parser()
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    hu_thresh = args.hu_threshold
    frac_thresh = args.fraction_threshold

    logging.info(f"Trimme leading slices: HU<{hu_thresh}, fraction>{frac_thresh}")
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(input_dir)
                   if f.endswith(".nii") or f.endswith(".nii.gz"))
    if not files:
        logging.warning("Keine NIfTI-Dateien gefunden.")
        sys.exit(0)

    total = 0
    for idx, fname in enumerate(files):
        in_path = os.path.join(input_dir, fname)
        logging.info(f"[{idx+1}/{len(files)}] Lade {in_path}")

        # 1) SITK => Numpy
        sitk_img = sitk.ReadImage(in_path)
        arr_3d = sitk.GetArrayFromImage(sitk_img)  # shape (D,H,W)

        # 2) Finde first lung-slice
        first_idx = find_first_lung_slice(arr_3d, hu_thresh, frac_thresh)
        if first_idx is None:
            logging.info(f"Keine Lunge gefunden.")
            continue

        # 3) Trim => arr_3d[first_idx:]
        trimmed_arr = arr_3d[first_idx:]
        logging.info(f"Trimming top {first_idx} slices => shape neu {trimmed_arr.shape}")

        # 4) SITK => .nii.gz
        out_img = sitk.GetImageFromArray(trimmed_arr)
        # Copy metadata (Spacing, Origin, Direction) 
        old_spacing = sitk_img.GetSpacing()  # (spacingZ, spacingY, spacingX)
        old_origin  = sitk_img.GetOrigin()
        old_direction = sitk_img.GetDirection()

        # Z-Verschiebung
        z_shift_mm = first_idx * old_spacing[2]  # Achtung: SITK-Spacings sind (z,y,x) oder (x,y,z) => muss gecheckt werden!!
        new_origin = list(old_origin)
        # verschieben origin in z-Richtung => typically new_origin[2] -= z_shift_mm  
        new_origin[2] = old_origin[2] + z_shift_mm  # oder minus -> kommt auf die Richtung an
        out_img.SetOrigin(tuple(new_origin))
        out_img.SetSpacing(old_spacing)
        out_img.SetDirection(old_direction)

        # 5) Speichern
        out_name = fname.replace(".nii", "_trimmed.nii").replace(".gz", "_trimmed.nii.gz")
        out_path = os.path.join(output_dir, out_name)
        sitk.WriteImage(out_img, out_path)
        logging.info(f"Gespeichert: {out_path}")

        total += 1

    logging.info(f"Fertig. {total} Volumes getrimmt.")

if __name__=="__main__":
    main()


# python3.11 preprocessing\trim_leading_slices.py --input_dir "D:\thesis_robert\xx_test" --output_dir "D:\thesis_robert\subset_v2_trimmed" --hu_threshold -300 --fraction_threshold 0.42