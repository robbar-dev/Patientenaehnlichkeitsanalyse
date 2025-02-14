import os
import argparse
import nibabel as nib
import numpy as np
from tqdm import tqdm

def remove_black_slices(volume_np):
    """
    Entfernt am Anfang und am Ende vollständig schwarze Slices aus den Serien.
    Das Skript erkennt vollständig schwarze Slices, indem es jeden Slice Zeile für Zeile (bzw. Pixel für Pixel) auf den Wert 0 überprüft und schaut, 
    ob alle Werte in diesem Slice tatsächlich exakt 0 sind. 

    Args:
        volume_np (np.ndarray): 3D-Array der Form (H, W, D).
                                Werte sind bereits normalisiert (0..1) oder in HU etc.

    Returns:
        np.ndarray: 3D-Array, in dem schwarze Rand-Slices entfernt wurden.
    """
    # volume_np.shape -> (H, W, D)
    H, W, D = volume_np.shape

    first_non_zero = None
    last_non_zero = None

    # Finde ersten Slice, der NICHT komplett schwarz ist
    for i in range(D):
        # slice_data -> (H, W)
        slice_data = volume_np[:, :, i]
        if not np.all(slice_data == 0):
            first_non_zero = i
            break

    # Finde letzten Slice, der NICHT komplett schwarz ist
    for i in reversed(range(D)):
        slice_data = volume_np[:, :, i]
        if not np.all(slice_data == 0):
            last_non_zero = i
            break

    # Falls ALLE Slices schwarz sind:
    if first_non_zero is None or last_non_zero is None:
        # Volumen komplett zurückgeben (nicht cropped)
        return volume_np

    # Schneide die Slices weg, die komplett schwarz sind
    volume_cropped = volume_np[:, :, first_non_zero:last_non_zero+1]
    return volume_cropped

def process_file(input_path, output_path):
    """
    Lädt eine NIfTI-Datei, entfernt schwarze Slices am Anfang und Ende, speichert das Ergebnis.
    """
    # Lade NIfTI mit nibabel
    nii = nib.load(input_path)
    volume = nii.get_fdata()  # numpy array, float64 standard
    affine = nii.affine       # Transformation/Orientierung
    header = nii.header       # NIfTI-Header

    # volume hat typischerweise (H, W, D)
    volume_cropped = remove_black_slices(volume)

    # Erzeuge neues NIfTI-Objekt
    new_nii = nib.Nifti1Image(volume_cropped, affine, header)

    # Speichern
    nib.save(new_nii, output_path)

def main():
    parser = argparse.ArgumentParser(
        description="Entfernt am Anfang oder Ende vollstaendig schwarze Slices in NIfTI-Dateien."
    )
    parser.add_argument("--input_dir", required=True, help="Eingabeverzeichnis mit NIfTI-Dateien")
    parser.add_argument("--output_dir", required=True, help="Ausgabeverzeichnis")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Durchsuche das Eingabeverzeichnis nach NIfTI-Dateien
    nifti_files = [
        f for f in os.listdir(input_dir)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ]

    if not nifti_files:
        print(f"Keine NIfTI-Dateien in {input_dir} gefunden.")
        return

    # Verarbeite jede Datei
    for filename in tqdm(nifti_files, desc="Verarbeite Dateien"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Prozess
        process_file(input_path, output_path)

if __name__ == "__main__":
    main()


# python preprocessing\black_slice_remover.py --input_dir "D:\thesis_robert\NLST_subset_v4_nifti_3mm_Voxel_with_black_slices" --output_dir "D:\thesis_robert\NLST_subset_v4_nifti_3mm_Voxel"

