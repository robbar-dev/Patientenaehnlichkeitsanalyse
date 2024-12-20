import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def visualize_with_slider(file_path):
    """
    Visualisiert eine NIfTI-Serie mit einem Schieberegler, um durch die Schichten zu scrollen.

    :param file_path: Pfad zur NIfTI-Serie
    """
    # NIfTI-Datei laden
    img = nib.load(file_path)
    data = img.get_fdata()
    
    # Initiale Schicht
    initial_slice = data.shape[2] // 2

    # Plot einrichten
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    slice_display = ax.imshow(data[:, :, initial_slice].T, cmap="gray", origin="lower")
    ax.set_title(f"Serie: {os.path.basename(file_path)}")
    ax.axis("off")

    # Schieberegler einrichten
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor="lightgray")
    slider = Slider(ax_slider, "Slice", 0, data.shape[2] - 1, valinit=initial_slice, valstep=1)

    # Update-Funktion für den Schieberegler
    def update(val):
        slice_idx = int(slider.val)
        slice_display.set_data(data[:, :, slice_idx].T)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

if __name__ == "__main__":
    input_folder = r"D:\thesis_robert\NLST_subset_v4_nifti_3mm_backup_outlier"
    
    # Liste aller NIfTI-Dateien im Ordner
    nifti_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
                   if f.endswith(".nii.gz") or f.endswith(".nii.gz.nii.gz")]
    
    if not nifti_files:
        print("Keine NIfTI-Dateien im angegebenen Ordner gefunden.")
    else:
        # Dateien einzeln visualisieren
        for file in nifti_files:
            print(f"Visualisiere Serie: {os.path.basename(file)}")
            visualize_with_slider(file)
