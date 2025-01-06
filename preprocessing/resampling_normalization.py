import argparse
import os
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    SaveImage,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Rotate90d,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
    Compose,
)
from monai.data import Dataset, DataLoader, pad_list_data_collate
import nibabel as nib

# Logging einrichten
logging.basicConfig(
    filename="resample_normalize.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def resample_and_normalize(
    input_dir,
    output_dir,
    target_spacing=(3.0, 3.0, 3.0),
    interpolation="trilinear",
    visualize=False,
    batch_size=4,
    num_workers=8
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # SaveImage Transform initialisieren
    save_transform = SaveImage(
        output_dir=output_dir,
        output_postfix="",
        output_ext=".nii.gz",
        output_dtype=np.float32,
        resample=False,
        separate_folder=False,
        print_log=True
    )

    # Transformationspipeline definieren
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="LPS"),
        Rotate90d(keys=["image"], k=2, spatial_axes=(0, 1)),  # Tisch unter den Patienten
        Spacingd(keys=["image"], pixdim=target_spacing, mode=interpolation),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000, a_max=400,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        EnsureTyped(keys=["image"], data_type="tensor")
    ])

    # Liste aller NIfTI-Dateien erstellen
    nifti_files = [
        {"image": os.path.join(input_dir, f)}
        for f in os.listdir(input_dir)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ]

    # Warnung ausgeben, wenn keine Dateien gefunden wurden
    if not nifti_files:
        logging.warning(f"Keine NIfTI-Dateien im Verzeichnis {input_dir} gefunden.")
        print(f"Warnung: Keine NIfTI-Dateien im Verzeichnis {input_dir} gefunden.")
        return

    # Liste der bereits verarbeiteten Dateien erstellen
    processed_filenames = set(os.listdir(output_dir))
    # Falls die Ausgabedateien die gleiche Erweiterung haben wie die Eingabedateien
    processed_files = set()
    for fname in processed_filenames:
        if fname.endswith(".nii") or fname.endswith(".nii.gz"):
            processed_files.add(fname)

    # Vorverarbeitung der Dateiliste: Überprüfen und Entfernen fehlerhafter und bereits verarbeiteter Dateien
    valid_nifti_files = []
    invalid_files = []
    skipped_files = []
    for file_dict in nifti_files:
        file_path = file_dict["image"]
        filename = os.path.basename(file_path)
        output_file_path = os.path.join(output_dir, filename)

        # Überprüfen, ob die Ausgabedatei bereits existiert
        if filename in processed_files:
            print(f"Datei {filename} bereits verarbeitet. Überspringe...")
            logging.info(f"Datei {filename} bereits verarbeitet. Überspringe...")
            skipped_files.append(filename)
            continue

        try:
            # Versuch, die Datei mit nibabel zu laden
            nib.load(file_path)
            valid_nifti_files.append(file_dict)
        except Exception as e:
            print(f"Fehler beim Laden der Datei {file_path}: {e}")
            logging.warning(f"Fehler beim Laden der Datei {file_path}: {e}")
            # PID aus dem Dateinamen extrahieren und zur Liste hinzufügen
            pid = filename
            invalid_files.append(pid)

    # Überprüfen, ob nach der Validierung noch Dateien übrig sind
    if not valid_nifti_files:
        logging.warning(f"Keine gültigen und unverarbeiteten NIfTI-Dateien im Verzeichnis {input_dir} gefunden.")
        print(f"Warnung: Keine gültigen und unverarbeiteten NIfTI-Dateien im Verzeichnis {input_dir} gefunden.")
        return

    # Dataset mit den validen Dateien erstellen
    dataset = Dataset(data=valid_nifti_files, transform=transforms)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=pad_list_data_collate
    )

    total_files = len(valid_nifti_files)
    processed_files_count = 0

    for batch_data in tqdm(loader, total=len(loader), desc="Progress:"):
        try:
            batch_size_actual = len(batch_data["image"])
            for i in range(batch_size_actual):
                image = batch_data["image"][i]
                # Metadaten aus dem MetaTensor abrufen
                meta = image.meta
                filename = os.path.basename(meta["filename_or_obj"])

                # Debugging-Ausgabe
                print(f"Verarbeite Datei: {filename}")
                print(f"Bildform: {image.shape}")
                print(f"Verfügbare Metadaten-Schlüssel: {list(meta.keys())}")

                # Daten speichern mit SaveImage
                save_transform(
                    img=image,
                    meta_data=meta,
                    filename=os.path.join(output_dir, filename)
                )
                print(f"Datei gespeichert: {os.path.join(output_dir, filename)}")

                # Optional: Visualisierung
                if visualize:
                    print("Visualisierung wird ausgeführt.")
                    depth = image.shape[-1]
                    slice_indices = [depth // 4, depth // 2, 3 * depth // 4]
                    for idx in slice_indices:
                        plt.imshow(image[0, :, :, idx], cmap="gray")
                        plt.title(f"Slice {idx}")
                        plt.show()

                processed_files_count += 1
        except Exception as e:
            logging.exception(f"Fehler bei der Verarbeitung eines Batches: {e}")
            print(f"Fehler bei der Verarbeitung: {e}")
            continue

    print(f"Insgesamt verarbeitete Dateien: {processed_files_count}/{total_files}")

    # Liste der fehlerhaften PIDs speichern
    if invalid_files or skipped_files:
        summary_path = os.path.join(output_dir, "processing_summary.txt")
        with open(summary_path, 'w') as f:
            if invalid_files:
                f.write("Nicht verarbeitete Dateien aufgrund von Fehlern:\n")
                for pid in invalid_files:
                    f.write(f"{pid}\n")
                f.write("\n")
            if skipped_files:
                f.write("Übersprungene Dateien (bereits verarbeitet):\n")
                for pid in skipped_files:
                    f.write(f"{pid}\n")
        print(f"Zusammenfassung der nicht verarbeiteten und übersprungenen Dateien gespeichert unter {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Resampling und Normalisierung von CT-Bildern")
    parser.add_argument("--input_dir", required=True, help="Pfad zu den Eingabedaten")
    parser.add_argument("--output_dir", required=True, help="Pfad zum Speichern der Ausgabedaten")
    parser.add_argument("--target_spacing", nargs=3, type=float, default=(3.0, 3.0, 3.0), help="Zielauflösung in mm")
    parser.add_argument("--interpolation", default="trilinear", help="Interpolationsmethode")
    parser.add_argument("--visualize", action="store_true", help="Visualisierung der Ergebnisse")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch-Größe für die Verarbeitung")
    parser.add_argument("--num_workers", type=int, default=8, help="Anzahl der Worker-Prozesse")
    args = parser.parse_args()

    resample_and_normalize(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_spacing=tuple(args.target_spacing),
        interpolation=args.interpolation,
        visualize=args.visualize,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()

# python preprocessing\resampling_normalization.py --input_dir "D:\thesis_robert\NLST_subset_v4_nifti_3mm_Voxel\validation_resampling_normalization\backup_outlier" --output_dir "D:\thesis_robert\NLST_subset_v4_nifti_3mm_Voxel"

