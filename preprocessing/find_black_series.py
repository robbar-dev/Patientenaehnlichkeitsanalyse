import os
import csv
import nibabel as nib
import numpy as np

INPUT_DIR = r"D:\thesis_robert\subset_v2_seg"
OUTPUT_CSV = r"D:\thesis_robert\black_series.csv"


def is_black_series(volume):
    """
    Prüft, ob eine NIfTI-Serie komplett schwarz ist (alle Werte 0).

    Args:
        volume (np.ndarray): 3D-Array des Volumens

    Returns:
        bool: True, wenn das Volumen komplett schwarz ist, sonst False
    """
    return np.max(volume) == 0


def find_black_series(input_dir, output_csv):
    """
    Durchsucht alle NIfTI-Dateien im Verzeichnis und speichert PIDs und Study-Years
    von komplett schwarzen Serien in einer CSV-Datei.

    Args:
        input_dir (str): Verzeichnis mit NIfTI-Dateien
        output_csv (str): Pfad zur Ausgabe-CSV-Datei

    Returns:
        None
    """
    black_series = []

    # Iteriere über alle Dateien im Verzeichnis
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".nii.gz"):
            continue

        file_path = os.path.join(input_dir, file_name)
        try:
            # Lade die NIfTI-Datei
            img = nib.load(file_path)
            volume = img.get_fdata()

            # Prüfe, ob die Serie schwarz ist
            if is_black_series(volume):
                # Extrahiere PID und Study Year aus dem Dateinamen
                parts = file_name.split("_")
                pid = parts[1] if len(parts) > 1 else "unknown"
                study_yr = parts[3] if len(parts) > 3 else "unknown"
                black_series.append((pid, study_yr))
        except Exception as e:
            print(f"Fehler beim Verarbeiten von {file_path}: {e}")

    # Schreibe die Ergebnisse in eine CSV-Datei
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["pid", "study_yr"])
        writer.writerows(black_series)

    print(f"Analyse abgeschlossen. Ergebnisse gespeichert in {output_csv}")


if __name__ == "__main__":
    find_black_series(INPUT_DIR, OUTPUT_CSV)
