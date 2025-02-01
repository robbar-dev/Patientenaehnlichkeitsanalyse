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
    AdjustContrastd,
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
    target_spacing=(1.5, 1.5, 1.5),
    interpolation="trilinear",
    visualize=False,
    batch_size=4,
    num_workers=8,
    a_min=-950,  # Feinanpassung für Emphyseme
    a_max=-500,  # Kontraste besser sichtbar
    gamma=1.5    # Erhöht den Kontrast
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
            a_min=a_min, a_max=a_max,  # Angepasste Normalisierung für Emphyseme
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        AdjustContrastd(keys=["image"], gamma=gamma),  # Kontrastanpassung
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

    # Dataset mit den validen Dateien erstellen
    dataset = Dataset(data=nifti_files, transform=transforms)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=pad_list_data_collate
    )

    total_files = len(nifti_files)
    processed_files_count = 0

    for batch_data in tqdm(loader, total=len(loader), desc="Progress:"):
        try:
            batch_size_actual = len(batch_data["image"])
            for i in range(batch_size_actual):
                image = batch_data["image"][i]
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

def main():
    parser = argparse.ArgumentParser(description="Resampling und Normalisierung von CT-Bildern")
    parser.add_argument("--input_dir", required=True, help="Pfad zu den Eingabedaten")
    parser.add_argument("--output_dir", required=True, help="Pfad zum Speichern der Ausgabedaten")
    parser.add_argument("--target_spacing", nargs=3, type=float, default=(1.5, 1.5, 1.5), help="Zielauflösung in mm")
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
        num_workers=args.num_workers,
        a_min=-200,  # Feinanpassung für Emphyseme
        a_max=1200,  # Kontraste besser sichtbar
        gamma=0.8   # Kontrast
    )

if __name__ == "__main__":
    main()

# python3.11 preprocessing\resampling_normalization.py --input_dir "D:\thesis_robert\NLST_subset_v6_nifti_unverarbeitet" --output_dir "D:\thesis_robert\NLST_subset_v6"

