import os
import argparse
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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

def find_matching_file(data_root, pid, study_yr):
    """
    Sucht eine Datei, die mit .nii.gz endet und die PID und Study Year enthält.

    Args:
        data_root (str): Wurzelverzeichnis der Dateien
        pid (str): Patient ID
        study_yr (str): Study Year

    Returns:
        str: Pfad zur gefundenen Datei oder None
    """
    for file in os.listdir(data_root):
        if file.endswith(".nii.gz") and f"pid_{pid}_study_yr_{study_yr}" in file:
            return os.path.join(data_root, file)
    return None

def visualize_series(df, data_root, num_patients):
    """
    Visualisiert verschiedene Klassen von NIfTI-Volumes interaktiv.

    Args:
        df (pd.DataFrame): DataFrame mit den Informationen [pid, study_yr, combination].
        data_root (str): Pfad zum Verzeichnis der NIfTI-Daten.
        num_patients (int): Anzahl der zu visualisierenden Patienten pro Klasse.
    """
    unique_combinations = df['combination'].unique()

    # Liste der NIfTI-Volumes für jede Kombination
    volumes = {}
    metadata = {}
    for combination in unique_combinations:
        patient_rows = df[df['combination'] == combination].head(num_patients)
        patient_volumes = []
        patient_metadata = []
        for _, row in patient_rows.iterrows():
            pid = row['pid']
            study_yr = row['study_yr']
            nifti_path = find_matching_file(data_root, pid, study_yr)
            if nifti_path:
                volume = load_nifti_volume(nifti_path)
                patient_volumes.append(volume)
                patient_metadata.append((pid, study_yr))
        volumes[combination] = patient_volumes
        metadata[combination] = patient_metadata

    # Visualisierung für verschiedene Kombinationen mit max. 2 Kombinationen pro Fenster
    current_combination_idx = 0
    total_combinations = len(unique_combinations)

    while current_combination_idx < total_combinations:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        sliders = []

        for i in range(2):
            if current_combination_idx >= total_combinations:
                axes[i].axis("off")
                continue

            combination = unique_combinations[current_combination_idx]
            if combination not in volumes or not volumes[combination]:
                axes[i].axis("off")
                current_combination_idx += 1
                continue

            volume = volumes[combination][0]  # Nur den ersten Patienten pro Kombination anzeigen
            pid, study_yr = metadata[combination][0]

            ax = axes[i]
            img = ax.imshow(volume[:, :, 0], cmap="gray")
            ax.set_title(f"Kombination: {combination} | PID: {pid}, Study Year: {study_yr}")
            ax.axis("off")

            # Slider direkt unter dem jeweiligen Bild
            slider_ax = plt.axes([0.15, 0.48 - i * 0.48, 0.7, 0.03])
            slider = Slider(slider_ax, f"Kombination {combination}", 0, volume.shape[2] - 1, valinit=0, valstep=1)
            sliders.append((slider, img, volume))

            current_combination_idx += 1

        def update(val):
            for slider, img, volume in sliders:
                slice_idx = int(slider.val)
                img.set_data(volume[:, :, slice_idx])
                fig.canvas.draw_idle()

        for slider, _, _ in sliders:
            slider.on_changed(update)

        plt.tight_layout()
        plt.pause(0.1)  # Blockiere kurz, um sicherzustellen, dass das Fenster gerendert wird
        plt.show()  # Fenster anzeigen und auf Schließen warten

def main():
    parser = argparse.ArgumentParser(description="Visualisierung von NIfTI-Serien basierend auf Klassenkombinationen.")
    parser.add_argument("--csv", required=True, help="Pfad zur CSV-Datei mit [pid, study_yr, combination].")
    parser.add_argument("--data_root", required=True, help="Pfad zum Wurzelverzeichnis der NIfTI-Daten.")
    parser.add_argument("--num_patients", type=int, default=5, help="Anzahl der Patientenserien pro Klasse zur Visualisierung.")
    args = parser.parse_args()

    # CSV-Datei mit korrektem Trennzeichen laden
    df = pd.read_csv(args.csv, sep=",")
    visualize_series(df, args.data_root, args.num_patients)

if __name__ == "__main__":
    main()


# python3.11 utils\test_model\visualization_series.py --csv "C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\classification_test\nlst_subset_v5_2classes.csv" --data_root "D:\thesis_robert\NLST_subset_v5_nifti_3mm_Voxel" --num_patients 4