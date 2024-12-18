import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def load_outlier_filenames(outlier_file):
    """
    Lädt die Seriennamen aus der Outlier-Textdatei.

    :param outlier_file: Pfad zur Outlier-Textdatei
    :return: Liste der Seriennamen
    """
    outlier_filenames = []
    with open(outlier_file, 'r') as f:
        for line in f:
            # Extrahiere den Dateinamen aus jeder Zeile
            if line.strip():  # Überspringe leere Zeilen
                parts = line.split(":")
                if len(parts) > 1:
                    filename = parts[1].strip().split(" ")[0]
                    outlier_filenames.append(filename)
    return list(set(outlier_filenames))  # Entferne Duplikate

def visualize_outliers(input_folder, outlier_filenames, images_per_window=9):
    """
    Visualisiert die Serien, die in der Outlier-Datei aufgeführt sind.

    :param input_folder: Pfad zu den NIfTI-Serien
    :param outlier_filenames: Liste der Seriennamen, die visualisiert werden sollen
    :param images_per_window: Anzahl der Bilder pro Fenster (max. 9 empfohlen)
    """
    # Verfügbare Dateien prüfen
    available_files = [f for f in os.listdir(input_folder) if f in outlier_filenames]

    if not available_files:
        print("Keine Dateien aus der Outlier-Liste im Eingabeordner gefunden.")
        return

    # Dateien in Gruppen aufteilen (max. images_per_window pro Fenster)
    num_windows = (len(available_files) + images_per_window - 1) // images_per_window
    for window in range(num_windows):
        start_idx = window * images_per_window
        end_idx = min(start_idx + images_per_window, len(available_files))
        current_files = available_files[start_idx:end_idx]

        # Matplotlib-Einstellungen für die Darstellung
        num_images = len(current_files)
        cols = 3  # Anzahl der Spalten
        rows = (num_images + cols - 1) // cols  # Anzahl der benötigten Zeilen
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()  # Achsen in 1D-Array umwandeln

        # Serien verarbeiten und anzeigen
        for i, file in enumerate(current_files):
            file_path = os.path.join(input_folder, file)
            print(f"Verarbeite Serie: {file}")
            
            # NIfTI-Datei laden und Daten extrahieren
            img = nib.load(file_path)
            data = img.get_fdata()

            # Ein Bild aus der Mitte der Serie auswählen (entlang der z-Achse)
            mid_slice = data.shape[2] // 2
            image_slice = data[:, :, mid_slice]

            # Bild darstellen
            axes[i].imshow(image_slice.T, cmap="gray", origin="lower")
            axes[i].set_title(f"Serie: {file}")
            axes[i].axis("off")  # Achsen ausblenden

        # Nicht verwendete Subplots ausblenden
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

# Hauptprogramm
if __name__ == "__main__":
    outlier_file = r"D:\thesis_robert\NLST_subset_v4_nifti_3mm_Voxel\validation_resampling_normalization\outliers_summary.txt"
    input_folder = r"D:\thesis_robert\NLST_subset_v4_nifti_3mm_Voxel"

    # Seriennamen aus der Outlier-Textdatei laden
    outlier_filenames = load_outlier_filenames(outlier_file)

    # Serien visualisieren
    visualize_outliers(input_folder, outlier_filenames, images_per_window=9)
