import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import logging

# Konfiguration
INPUT_DIR = r"D:\thesis_robert\NLST_subset_v5_SEG_NORM_nifti_1_5mm_Voxel"
OUTPUT_DIR = r"D:\thesis_robert\NLST_subset_v5_SEG_NORM_nifti_1_5mm_Voxel_cropped"
TOLERANCE = 3  # Anzahl der zusätzlichen Pixel um die Lunge herum

# Logging konfigurieren
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

def calculate_initial_bounding_box(volume):
    """
    Berechnet eine enge Bounding Box basierend auf Nicht-Schwarz-Pixeln.

    Args:
        volume (np.ndarray): 3D-Array der NIfTI-Daten

    Returns:
        tuple: (min_row, max_row, min_col, max_col)
    """
    # Projektion über alle Slices
    non_zero_projection = np.max(volume > 0, axis=2)

    # Minimaler Bereich ohne Schwarz
    rows = np.any(non_zero_projection, axis=1)
    cols = np.any(non_zero_projection, axis=0)

    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    return min_row, max_row, min_col, max_col

def calculate_final_bounding_box(volume, tolerance):
    """
    Berechnet die Bounding Box der Lunge mit einer Toleranz basierend auf einer mittleren Slice.

    Args:
        volume (np.ndarray): 3D-Array der NIfTI-Daten
        tolerance (int): Toleranz in Pixeln um die Bounding Box herum

    Returns:
        tuple: (min_row, max_row, min_col, max_col)
    """
    middle_slice_idx = volume.shape[2] // 2
    middle_slice = volume[:, :, middle_slice_idx] > 0

    rows = np.any(middle_slice, axis=1)
    cols = np.any(middle_slice, axis=0)

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
    # Sicherstellen, dass das Ausgabe-Verzeichnis existiert
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for file_name in os.listdir(INPUT_DIR):
        if not file_name.endswith(".nii.gz"):
            continue

        input_path = os.path.join(INPUT_DIR, file_name)
        output_path = os.path.join(OUTPUT_DIR, file_name)

        logging.info(f"Verarbeite Datei: {input_path}")

        try:
            # Volume laden
            volume, affine = load_nifti_volume(input_path)

            # Erste Bounding Box berechnen, um schwarzen Rand zu entfernen
            initial_bounding_box = calculate_initial_bounding_box(volume)
            cropped_volume = crop_volume(volume, initial_bounding_box)

            # Finale Bounding Box basierend auf der mittleren Slice berechnen
            final_bounding_box = calculate_final_bounding_box(cropped_volume, TOLERANCE)
            final_cropped_volume = crop_volume(cropped_volume, final_bounding_box)

            # Zuschnitt speichern
            save_nifti_volume(final_cropped_volume, affine, output_path)
            logging.info(f"Gespeichert: {output_path}")

        except Exception as e:
            logging.error(f"Fehler bei der Verarbeitung von {file_name}: {e}")

if __name__ == "__main__":
    main()
