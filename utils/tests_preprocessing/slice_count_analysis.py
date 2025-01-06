import os
import nibabel as nib

def analyze_nifti_slices(input_folder):
    """
    Analysiere die Slice-Anzahl der NIfTI-Dateien in einem Ordner.

    Args:
        input_folder (str): Pfad zum Ordner mit den NIfTI-Dateien.

    Returns:
        dict: Analyseergebnisse mit durchschnittlicher, maximaler und minimaler Slice-Anzahl.
    """
    slice_counts = []

    # Iteriere über alle Dateien im Ordner
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                file_path = os.path.join(root, file)

                # Lade die NIfTI-Datei
                try:
                    nifti_img = nib.load(file_path)
                    data = nifti_img.get_fdata()

                    # Extrahiere die Anzahl der Slices (Dritte Dimension)
                    slice_count = data.shape[2]
                    slice_counts.append(slice_count)
                except Exception as e:
                    print(f"Fehler beim Verarbeiten von {file_path}: {e}")

    if not slice_counts:
        print("Keine NIfTI-Dateien gefunden oder keine gültigen Dateien verarbeitet.")
        return None

    # Berechnung der Analyseergebnisse
    average_slices = sum(slice_counts) / len(slice_counts)
    max_slices = max(slice_counts)
    min_slices = min(slice_counts)

    return {
        "average_slices": average_slices,
        "max_slices": max_slices,
        "min_slices": min_slices
    }

if __name__ == "__main__":
    input_folder = r"D:\thesis_robert\NLST_subset_v4_nifti_3mm_Voxel"

    if not os.path.exists(input_folder):
        print("Der angegebene Ordner existiert nicht.")
    else:
        analysis_results = analyze_nifti_slices(input_folder)

        if analysis_results:
            print("\nAnalyse der Slice-Anzahl der NIfTI-Dateien:")
            print(f"Durchschnittliche Slice-Anzahl: {analysis_results['average_slices']:.2f}")
            print(f"Maximale Slice-Anzahl: {analysis_results['max_slices']}")
            print(f"Minimale Slice-Anzahl: {analysis_results['min_slices']}")
