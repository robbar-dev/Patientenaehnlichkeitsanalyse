import os
import random
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def visualize_sample_series(input_folder, total_samples, images_per_window=9):
    """
    Visualisiert eine Stichprobe von NIfTI-Serien. Aus jeder Serie wird ein Bild aus der Mitte extrahiert
    und in mehreren Fenstern dargestellt (max. 9 Bilder pro Fenster).

    :param input_folder: Pfad zum Ordner mit den NIfTI-Serien
    :param total_samples: Gesamtzahl der Stichproben
    :param images_per_window: Anzahl der Bilder pro Fenster (max. 9 empfohlen)
    """
    # Alle Dateien im Input-Ordner auflisten und zufällige Stichprobe ziehen
    all_files = [f for f in os.listdir(input_folder) if f.endswith('.nii.gz') or f.endswith('.nii.gz.nii.gz')]
    sample_files = random.sample(all_files, min(total_samples, len(all_files)))

    # Dateien in Gruppen aufteilen (max. images_per_window pro Fenster)
    num_windows = (len(sample_files) + images_per_window - 1) // images_per_window
    for window in range(num_windows):
        start_idx = window * images_per_window
        end_idx = min(start_idx + images_per_window, len(sample_files))
        current_files = sample_files[start_idx:end_idx]

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
    input_folder = r"D:\thesis_robert\NLST_subset_v5_seg_nifti_3mm_Voxel"
    visualize_sample_series(input_folder, total_samples=90, images_per_window=9)
