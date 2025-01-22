import os
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random

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
    return volume

def visualize_sample(nifti_files, sample_size):
    """
    Visualisiert eine Stichprobe von NIfTI-Dateien mit bis zu 2 Serien pro Fenster.

    Args:
        nifti_files (list): Liste der Pfade zu NIfTI-Dateien.
        sample_size (int): Anzahl der zu visualisierenden Stichproben.
    """
    if sample_size > len(nifti_files):
        print("Warnung: Stichprobe größer als Anzahl der verfügbaren Dateien. Reduziere Stichprobengröße.")
        sample_size = len(nifti_files)

    # Zufällige Auswahl von Dateien
    sampled_files = random.sample(nifti_files, sample_size)

    current_idx = 0
    total_files = len(sampled_files)

    while current_idx < total_files:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.subplots_adjust(hspace=0.3)

        sliders = []
        displayed_images = []

        for i in range(2):
            if current_idx >= total_files:
                axes[i].axis("off")
                continue

            file_path = sampled_files[current_idx]
            volume = load_nifti_volume(file_path)

            ax = axes[i]
            img = ax.imshow(volume[:, :, 0], cmap="gray")
            ax.set_title(f"{os.path.basename(file_path)}", fontsize=10)
            ax.axis("off")

            slider_ax = fig.add_axes([0.15, 0.48 - i * 0.48, 0.7, 0.03])
            slider = Slider(slider_ax, f"Slice {i+1}", 0, volume.shape[2] - 1, valinit=0, valstep=1)

            sliders.append((slider, img, volume))
            displayed_images.append((img, volume))
            current_idx += 1

        def update(val):
            for slider, img, volume in sliders:
                slice_idx = int(slider.val)
                img.set_data(volume[:, :, slice_idx])
            fig.canvas.draw_idle()

        for slider, _, _ in sliders:
            slider.on_changed(update)

        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualisierung von NIfTI-Stichproben.")
    parser.add_argument("--data_root", required=True, help="Pfad zum Verzeichnis der NIfTI-Dateien.")
    parser.add_argument("--sample_size", type=int, default=4, help="Anzahl der zu visualisierenden Stichproben.")
    args = parser.parse_args()

    # Alle NIfTI-Dateien im Verzeichnis finden
    nifti_files = [os.path.join(args.data_root, f) for f in os.listdir(args.data_root) if f.endswith(".nii.gz")]

    if not nifti_files:
        print("Keine NIfTI-Dateien im angegebenen Verzeichnis gefunden.")
        return

    visualize_sample(nifti_files, args.sample_size)

if __name__ == "__main__":
    main()


# python3.11 utils\tests_preprocessing\visu_seg_help.py --data_root "D:\thesis_robert\Segmentation_Test\V2_unverarbeitet_Data\testdata_v2_unverarbeitet_seg" --sample_size 10
