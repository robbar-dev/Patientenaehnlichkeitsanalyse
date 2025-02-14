import os
import nibabel as nib
import numpy as np
import logging
from tqdm import tqdm 

INPUT_DIR = r"D:\thesis_robert\NLST_subset_v5_SEG_NORM_nifti_1_5mm_Voxel"
OUTPUT_DIR = r"D:\thesis_robert\NLST_subset_v5_SEG_NORM_nifti_1_5mm_Voxel_cropped_5_toleranz"
TOLERANCE = 5  # Anzahl der zusätzlichen Pixel um die Lunge herum -> um black slices zu vermeiden

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_nifti_volume(filepath):
    """
    Lädt ein NIfTI-Volume aus einer Datei.

    Args:
        filepath (str): Pfad zur NIfTI-Datei

    Returns:
        np.ndarray: 3D-Array der NIfTI-Daten
    """
    img = nib.load(filepath)
    volume = img.get_fdata()
    affine = img.affine
    return volume, affine

def save_nifti_volume(volume, affine, output_path):
    """
    Speichert ein NIfTI-Volume in einer Datei.

    Args:
        volume (np.ndarray): 3D-Array der NIfTI-Daten
        affine (np.ndarray): Affine-Transformation
        output_path (str): Zielpfad

    Returns:
        None
    """
    nifti_img = nib.Nifti1Image(volume, affine)
    nib.save(nifti_img, output_path)

def calculate_bounding_box(volume, tolerance):
    """
    Berechnet die Bounding Box des Volumens basierend auf mehreren Slices.

    Args:
        volume (np.ndarray): 3D-Array der NIfTI-Daten
        tolerance (int): Toleranz in Pixeln um die Bounding Box herum

    Returns:
        tuple: (min_row, max_row, min_col, max_col)
    """
    # Summiere über alle Slices, um die maximalen Bereiche zu erfassen
    projection = np.max(volume, axis=2) > 0  # Maximalprojektion über die Z-Achse

    # Finde minimale und maximale Koordinaten
    rows = np.any(projection, axis=1)
    cols = np.any(projection, axis=0)

    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    # Toleranz hinzufügen
    min_row = max(0, min_row - tolerance)
    max_row = min(volume.shape[0], max_row + tolerance)
    min_col = max(0, min_col - tolerance)
    max_col = min(volume.shape[1], max_col + tolerance)

    return min_row, max_row, min_col, max_col

def crop_volume(volume, bounding_box):
    """
    Schneidet das Volume basierend auf der Bounding Box zu.

    Args:
        volume (np.ndarray): 3D-Array der NIfTI-Daten
        bounding_box (tuple): (min_row, max_row, min_col, max_col)

    Returns:
        np.ndarray: Das zugeschnittene Volume
    """
    min_row, max_row, min_col, max_col = bounding_box
    cropped_volume = volume[min_row:max_row, min_col:max_col, :]
    return cropped_volume

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Liste aller Dateien im Eingabeverzeichnis
    files = [file for file in os.listdir(INPUT_DIR) if file.endswith(".nii.gz")]

    # Fortschrittsanzeige
    for file_name in tqdm(files, desc="Verarbeitung von Dateien", unit="Datei"):
        input_path = os.path.join(INPUT_DIR, file_name)
        output_path = os.path.join(OUTPUT_DIR, file_name)

        logging.info(f"Verarbeite Datei: {input_path}")

        try:
            # Volume laden
            volume, affine = load_nifti_volume(input_path)

            # Bounding Box berechnen basierend auf allen Slices
            bounding_box = calculate_bounding_box(volume, TOLERANCE)

            # Volume zuschneiden
            cropped_volume = crop_volume(volume, bounding_box)

            # Zuschnitt speichern
            save_nifti_volume(cropped_volume, affine, output_path)
            logging.info(f"Gespeichert: {output_path}")

        except Exception as e:
            logging.error(f"Fehler bei der Verarbeitung von {file_name}: {e}")

if __name__ == "__main__":
    main()
