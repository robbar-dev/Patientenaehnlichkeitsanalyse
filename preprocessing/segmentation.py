import os
import sys
import argparse
import logging

import SimpleITK as sitk
import numpy as np

# pip install lungmask
import lungmask

logging.basicConfig(level=logging.INFO)

"""
------------------------
Dieses Skript wendet das 'lungmask'-Package an, um CT-Volumes automatisch
zu segmentieren (Lunge vs. Nicht-Lunge).

Vorgehen:
1) Liest .nii oder .nii.gz Dateien aus input_dir mit SimpleITK.
2) Wendet lungmask.apply(...) an, das ein pretrained Model (z.B. 'R231') nutzt.
3) Speichert die binäre Lungenmaske (0=Hintergrund,1=Lunge) in output_dir.
4) Optional: Speichert das maskierte Volumen (Lunge * Original) per --save_masked_volume.

Beispielaufruf:
  python lungmask_segmentation.py \
    --input_dir "D:/thesis_robert/CT_preprocessed_1.5mm" \
    --output_dir "D:/thesis_robert/CT_lung_segmented" \
    --modelname "R231" \
    --save_masked_volume

Siehe GitHub: https://github.com/JoHof/lungmask
"""

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
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    modelname = args.modelname
    save_masked = args.save_masked_volume

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Starte Lungensegmentierung: input_dir={input_dir}, model={modelname}")

    # 1) Liste aller CT-Dateien
    ct_files = sorted([
        f for f in os.listdir(input_dir)
        if (f.endswith(".nii") or f.endswith(".nii.gz"))
    ])
    if not ct_files:
        logging.warning(f"Keine NIfTI-Dateien in {input_dir} gefunden.")
        sys.exit(0)

    # 2) Lade das pretrained Model aus lungmask
    #    -> lungmask.get_model(arch, modelname) (arch='unet', 'resunet', ...)
    #    Standard: arch='unet', modelname='R231'
    #    Alternativ: 'R231CovidWeb'
    logging.info(f"Lade pretrained lungmask Model: arch='unet', model='{modelname}'")
    model = lungmask.get_model('unet', modelname)
    # model ist ein torch.nn.Module -> kann auf GPU verschoben werden
    # lungmask selbst regelt GPU vs CPU. Du kannst aber unten "apply(image, model, force_cpu=...)"

    for idx, ct_filename in enumerate(ct_files):
        ct_path = os.path.join(input_dir, ct_filename)
        logging.info(f"[{idx+1}/{len(ct_files)}] Lade CT-Volume: {ct_path}")

        # 3) CT laden via SimpleITK
        image_sitk = sitk.ReadImage(ct_path)
        # 4) Wende lungmask.apply(...) an
        #    => Dies gibt ein numpy-Array [H,W,D] (Maske) zurück mit 0/1
        logging.info("Wende lungmask.apply(...) an, bitte warten...")
        mask_arr = lungmask.apply(image_sitk, model)  # shape (z,y,x) => SITK => (D,H,W)

        # 5) mask_arr in SITK konvertieren
        mask_sitk = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
        mask_sitk.CopyInformation(image_sitk)  # spacing, origin, direction übernehmen

        # 6) Speichern der Maske
        out_maskname = ct_filename.replace(".nii","_lungmask.nii").replace(".gz","_lungmask.nii.gz")
        mask_path = os.path.join(output_dir, out_maskname)
        sitk.WriteImage(mask_sitk, mask_path)
        logging.info(f"Maske gespeichert => {mask_path}")

        # 7) Optional: Maske * Original => maskiertes Volume
        if save_masked:
            logging.info("Erstelle maskiertes Volume (Maske * Original).")
            # SITK => Konvertieren in NumPy => multiply => back to SITK
            original_arr = sitk.GetArrayFromImage(image_sitk)  # (D,H,W)
            masked_arr = original_arr * mask_arr  # => nur Lunge
            masked_sitk = sitk.GetImageFromArray(masked_arr)
            masked_sitk.CopyInformation(image_sitk)

            masked_outname = ct_filename.replace(".nii","_lungmaskedvol.nii").replace(".gz","_lungmaskedvol.nii.gz")
            masked_path = os.path.join(output_dir, masked_outname)
            sitk.WriteImage(masked_sitk, masked_path)
            logging.info(f"Maskiertes Volumen gespeichert => {masked_path}")

    logging.info("FERTIG. Alle CTs wurden segmentiert.")


if __name__ == "__main__":
    main()

# python3.11 preprocessing\segmentation.py --input_dir "D:\thesis_robert\NLST_subset_v5_nifti_3mm_Voxel" --output_dir "D:\thesis_robert\NLST_subset_v5_nifti_3mm_Voxel_segmented" --modelname "R231" --save_masked_volume
