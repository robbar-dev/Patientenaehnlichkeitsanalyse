"""

Wird nicht merh benötigt. 
Dachte das löst das black slice Problem, wurde aber über cropper gelöst. 

"""

# """
# Min-Max Normalisierung + Umorientierung aller .nii/.nii.gz in einem Ordner,
# sodass volume[:,:,0] (axiale Slice) direkt sichtbar wird.

# Inklusive Debug-Ausgaben zu: Shape, min/max/mean, Histogram,
# und Stichproben einzelner Slices.
# """

# import os
# import sys
# import argparse
# import logging
# import numpy as np
# import nibabel as nib

# def get_parser():
#     parser = argparse.ArgumentParser(description="Min–Max Normalisierung + kanonische Orientierung (RAS) für NIfTI-Dateien.")
#     parser.add_argument("--input_dir", required=True, help="Pfad zum Verzeichnis mit den Eingabe-.nii / .nii.gz")
#     parser.add_argument("--output_dir", required=True, help="Ausgabeverzeichnis für die normalisierten NIfTI-Dateien")
#     parser.add_argument("--epsilon", type=float, default=1e-7, help="kleiner Wert, um Division durch 0 zu vermeiden")
#     return parser

# def minmax_normalize_volume(volume_np, eps=1e-7):
#     """
#     volume_np: np.ndarray (z.B. (H,W,D))
#     => returns volume_np skaliert auf [0..1]
#     """
#     vmin = volume_np.min()
#     vmax = volume_np.max()
#     rng = vmax - vmin
#     if rng < eps:
#         # Falls das Volume quasi konstant ist, => alles auf 0
#         return np.zeros_like(volume_np, dtype=np.float32)
#     else:
#         return (volume_np - vmin) / rng

# def main():
#     parser = get_parser()
#     args = parser.parse_args()

#     input_dir = args.input_dir
#     output_dir = args.output_dir
#     eps = args.epsilon

#     os.makedirs(output_dir, exist_ok=True)
#     # Level auf DEBUG, damit wir alle Logs sehen
#     logging.basicConfig(level=logging.DEBUG)

#     files = [f for f in os.listdir(input_dir) if (f.endswith(".nii") or f.endswith(".nii.gz"))]
#     if not files:
#         logging.warning(f"Keine .nii/.nii.gz Dateien in {input_dir}")
#         sys.exit(0)

#     processed = 0
#     for i, fname in enumerate(files):
#         in_path = os.path.join(input_dir, fname)
#         logging.info(f"\n=== [{i+1}/{len(files)}] Lade NIfTI: {in_path} ===")
#         try:
#             # 1) Laden und in kanonische Orientierung wandeln
#             img_orig = nib.load(in_path)
#             logging.debug(f"[{fname}] Original shape: {img_orig.shape}")
#             img_can = nib.as_closest_canonical(img_orig)

#             # 2) Hole Daten (float32) und minmax-normalisieren
#             volume_np = img_can.get_fdata(dtype=np.float32)

#             # Debug-Ausgabe vor der Normalisierung
#             logging.debug(f"[{fname}] Kanonische shape: {volume_np.shape}")
#             logging.debug(f"[{fname}] Kanonische min={volume_np.min():.6f}, max={volume_np.max():.6f}, mean={volume_np.mean():.6f}")

#             volume_np = minmax_normalize_volume(volume_np, eps)

#             # Debug-Ausgaben nach Min-Max
#             logging.debug(f"[{fname}] Nach Normalisierung: shape={volume_np.shape}")
#             logging.debug(f"[{fname}] min={volume_np.min():.6f}, max={volume_np.max():.6f}, mean={volume_np.mean():.6f}")

#             # Kleines Histogramm
#             vals = volume_np.flatten()
#             hist, bin_edges = np.histogram(vals, bins=20, range=(0,1))
#             logging.debug(f"[{fname}] Histogram (20 Bins) in [0..1]:")
#             for b, h in zip(bin_edges, hist):
#                 logging.debug(f"   BinEdge={b:.3f}: count={h}")

#             # Falls 3D-Volume, schauen wir mal ein paar Slices an
#             if volume_np.ndim == 3:
#                 slice_indices = [0, 10, 20, 30, 40]
#                 zmax = volume_np.shape[2]
#                 for s in slice_indices:
#                     if s < zmax:
#                         slc = volume_np[:, :, s]
#                         logging.debug(f"[{fname}] Slice {s} => min={slc.min():.6f}, max={slc.max():.6f}, mean={slc.mean():.6f}")
#                     else:
#                         logging.debug(f"[{fname}] Slice {s} nicht vorhanden (zmax={zmax}).")
#             else:
#                 logging.debug(f"[{fname}] Kein 3D-Volume => skip Slice-Debug. ndims={volume_np.ndim}")

#             # 3) Neues Nifti mit kanonischer Affine & Header
#             norm_img = nib.Nifti1Image(volume_np, img_can.affine, img_can.header)

#             # 4) Output-Dateinamen
#             out_name = fname.replace(".nii","_minmax.nii").replace(".gz","_minmax.nii.gz")
#             out_path = os.path.join(output_dir, out_name)
#             nib.save(norm_img, out_path)
#             logging.info(f"[{fname}] Gespeichert => {out_path}")
#             processed += 1

#         except Exception as e:
#             logging.error(f"Fehler bei Datei {fname}: {e}")
#             continue

#     logging.info(f"Fertig. Insgesamt {processed}/{len(files)} Volumes normalisiert.")

# if __name__ == "__main__":
#     main()


# # python3.11 preprocessing\segmentation\min_max_normalization.py --input_dir "D:\thesis_robert\test_data_folder\black_series" --output_dir "D:\thesis_robert\test_data_folder\black_series"