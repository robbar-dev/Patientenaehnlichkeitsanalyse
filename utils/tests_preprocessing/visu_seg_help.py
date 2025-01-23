import os
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random

# Konfiguration
DATA_PATH = r"D:\thesis_robert\NLST_subset_v5_seg_nifti_1_5mm_Voxel"
SAMPLE_SIZE = 5 

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

def visualize_sampled_nifti_series(data_path, sample_size):
    """
    Visualisiert eine Stichprobe von NIfTI-Serien in einem Verzeichnis.

    Args:
        data_path (str): Pfad zum Verzeichnis der NIfTI-Dateien.
        sample_size (int): Anzahl der zu visualisierenden zufälligen Serien.

    Returns:
        None
    """
    # Alle NIfTI-Dateien im Verzeichnis auflisten
    nifti_files = [f for f in os.listdir(data_path) if f.endswith(".nii.gz")]

    if len(nifti_files) == 0:
        print("Keine NIfTI-Dateien im Verzeichnis gefunden.")
        return

    if sample_size > len(nifti_files):
        print("Warnung: Stichprobengröße ist größer als die Anzahl der verfügbaren Dateien.")
        sample_size = len(nifti_files)

    # Zufällige Auswahl von Dateien
    sampled_files = random.sample(nifti_files, sample_size)

    for file_name in sampled_files:
        file_path = os.path.join(data_path, file_name)
        print(f"Lade Datei: {file_path}")

        try:
            # NIfTI-Datei laden
            volume = load_nifti_volume(file_path)
            print("Shape:", volume.shape, "Min:", volume.min(), "Max:", volume.max())

            # Initiales Setup
            fig, ax = plt.subplots()
            plt.subplots_adjust(bottom=0.25)
            img_plot = ax.imshow(volume[:, :, 0].T,
                     cmap="gray",
                     origin="lower",
                     vmin=0,
                     vmax=1)

            ax.set_title(f"{file_name} - Slice 0")

            # Slider hinzufügen
            ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
            slider = Slider(ax_slider, "Slice", 0, volume.shape[2] - 1, valinit=0, valstep=1)

            def update(val):
                slice_idx = int(slider.val)
                slice_data = volume[:, :, slice_idx].T
                img_plot.set_data(slice_data)
                ax.set_title(f"{file_name} - Slice {slice_idx}")
                fig.canvas.draw_idle()

            slider.on_changed(update)

            plt.show()

        except Exception as e:
            print(f"Fehler beim Laden oder Visualisieren von {file_name}: {e}")

if __name__ == "__main__":
    visualize_sampled_nifti_series(DATA_PATH, SAMPLE_SIZE)



# python3.11 utils\tests_preprocessing\visu_seg_help.py --data_root "D:\thesis_robert\xx_test" --sample_size 10
