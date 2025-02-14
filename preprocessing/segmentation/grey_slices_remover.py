import os
import nibabel as nib
import numpy as np
import logging

DATA_PATH = r"D:\thesis_robert\subset_v5_seg"
LOG_FILE = r"D:\thesis_robert\remove_gray_slices.log"


logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

def remove_gray_slices(data_path):
    """
    Entfernt komplett graue Slices am Ende jeder Serie aus den NIfTI-Dateien.

    Args:
        data_path (str): Verzeichnis mit NIfTI-Dateien.

    Returns:
        None
    """
    logging.info(f"Starte Verarbeitung im Verzeichnis: {data_path}")

    for file_name in os.listdir(data_path):
        if not file_name.endswith(".nii.gz"):
            continue

        file_path = os.path.join(data_path, file_name)
        logging.info(f"Verarbeite Datei: {file_path}")

        try:
            # NIfTI-Datei laden
            img = nib.load(file_path)
            volume = img.get_fdata()
            affine = img.affine

            # Prüfe graue Slices
            unique_gray_values = []
            keep_slices = []

            for z in range(volume.shape[2]):
                slice_data = volume[:, :, z]
                if np.all(slice_data == slice_data[0, 0]):
                    unique_gray_values.append(slice_data[0, 0])
                else:
                    keep_slices.append(z)

            # Entferne graue Slices am Ende der Serie
            if keep_slices:
                first_keep = min(keep_slices)
                last_keep = max(keep_slices) + 1
                cleaned_volume = volume[:, :, first_keep:last_keep]

                # Speichern der bereinigten Datei
                cleaned_img = nib.Nifti1Image(cleaned_volume, affine)
                nib.save(cleaned_img, file_path)
                logging.info(f"Graue Slices entfernt. Gespeichert: {file_path}")
            else:
                logging.warning(f"Keine relevanten Slices in {file_name}. Datei wird nicht geändert.")

        except Exception as e:
            logging.error(f"Fehler beim Verarbeiten von {file_name}: {e}")

    logging.info("Verarbeitung abgeschlossen.")

if __name__ == "__main__":
    remove_gray_slices(DATA_PATH)
