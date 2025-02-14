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
    ScaleIntensityRanged,
    EnsureTyped,
    Compose,
)
from monai.data import Dataset, DataLoader, pad_list_data_collate
import nibabel as nib

logging.basicConfig(
    filename="normalize.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def normalize_intensities(
    input_dir,
    output_dir,
    a_min=-1000,
    a_max=400,
    b_min=0.0,
    b_max=1.0,
    clip=True,
    visualize=False,
    batch_size=4,
    num_workers=8
):
    """
    Lädt bereits resampelte (oder originale) Dateien aus input_dir,
    führt ggf. Orientation/Rotation aus und normalisiert die Intensitäten
    via ScaleIntensityRanged. Speichert das Ergebnis in output_dir.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_transform = SaveImage(
        output_dir=output_dir,
        output_postfix="",
        output_ext=".nii.gz",
        output_dtype=np.float32,
        resample=False,
        separate_folder=False,
        print_log=True
    )

    # Nur Intensitätsnormalisierung:
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=a_min, a_max=a_max,
            b_min=b_min, b_max=b_max,
            clip=clip
        ),
        EnsureTyped(keys=["image"], data_type="tensor")
    ])

    nifti_files = [
        {"image": os.path.join(input_dir, f)}
        for f in os.listdir(input_dir)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ]

    if not nifti_files:
        logging.warning(f"Keine NIfTI-Dateien im Verzeichnis {input_dir} gefunden.")
        print(f"Warnung: Keine NIfTI-Dateien im Verzeichnis {input_dir} gefunden.")
        return

    processed_filenames = set(os.listdir(output_dir))
    processed_files = set()
    for fname in processed_filenames:
        if fname.endswith(".nii") or fname.endswith(".nii.gz"):
            processed_files.add(fname)

    valid_nifti_files = []
    invalid_files = []
    skipped_files = []
    for file_dict in nifti_files:
        file_path = file_dict["image"]
        filename = os.path.basename(file_path)

        if filename in processed_files:
            print(f"Datei {filename} bereits verarbeitet. Überspringe...")
            logging.info(f"Datei {filename} bereits verarbeitet. Überspringe...")
            skipped_files.append(filename)
            continue

        try:
            nib.load(file_path)
            valid_nifti_files.append(file_dict)
        except Exception as e:
            print(f"Fehler beim Laden der Datei {file_path}: {e}")
            logging.warning(f"Fehler beim Laden der Datei {file_path}: {e}")
            invalid_files.append(filename)

    if not valid_nifti_files:
        logging.warning(f"Keine gültigen und unverarbeiteten NIfTI-Dateien gefunden.")
        print(f"Warnung: Keine gültigen und unverarbeiteten NIfTI-Dateien gefunden.")
        return

    dataset = Dataset(data=valid_nifti_files, transform=transforms)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=pad_list_data_collate
    )

    total_files = len(valid_nifti_files)
    processed_files_count = 0

    for batch_data in tqdm(loader, total=len(loader), desc="Normalize Progress:"):
        try:
            batch_size_actual = len(batch_data["image"])
            for i in range(batch_size_actual):
                image = batch_data["image"][i]
                meta = image.meta
                filename = os.path.basename(meta["filename_or_obj"])

                print(f"Normalize: Verarbeite Datei: {filename}")
                print(f"Bildform nach Normalisierung: {image.shape}")

                save_transform(
                    img=image,
                    meta_data=meta,
                    filename=os.path.join(output_dir, filename)
                )
                print(f"Datei gespeichert: {os.path.join(output_dir, filename)}")

                if visualize:
                    depth = image.shape[-1]
                    slice_indices = [depth // 4, depth // 2, 3 * depth // 4]
                    for idx in slice_indices:
                        plt.imshow(image[0, :, :, idx], cmap="gray")
                        plt.title(f"Slice {idx} (Normalized)")
                        plt.show()

                processed_files_count += 1
        except Exception as e:
            logging.exception(f"Fehler bei der Verarbeitung eines Batches: {e}")
            print(f"Fehler bei der Verarbeitung: {e}")
            continue

    print(f"Insgesamt verarbeitete Dateien (Normalization): {processed_files_count}/{total_files}")

    if invalid_files or skipped_files:
        summary_path = os.path.join(output_dir, "normalize_summary.txt")
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
        print(f"Normalize-Zusammenfassung gespeichert unter {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Normalisierung von CT-Bildern (ohne Resampling)")
    parser.add_argument("--input_dir", required=True, help="Pfad zu den Eingabedaten (z.B. bereits resampelt)")
    parser.add_argument("--output_dir", required=True, help="Pfad zum Speichern der Ausgabe (normalized)")
    parser.add_argument("--a_min", type=float, default=-1000.0, help="untere Intensitätsgrenze (HU)")
    parser.add_argument("--a_max", type=float, default=400.0, help="obere Intensitätsgrenze (HU)")
    parser.add_argument("--b_min", type=float, default=0.0, help="unteres Normalisierungslevel")
    parser.add_argument("--b_max", type=float, default=1.0, help="oberes Normalisierungslevel")
    parser.add_argument("--clip", action="store_true", help="Clip die Werte (Standard: True)")
    parser.add_argument("--visualize", action="store_true", help="Visualisierung einiger Slices")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch-Größe")
    parser.add_argument("--num_workers", type=int, default=8, help="Anzahl Worker")
    args = parser.parse_args()

    normalize_intensities(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        a_min=args.a_min,
        a_max=args.a_max,
        b_min=args.b_min,
        b_max=args.b_max,
        clip=args.clip,
        visualize=args.visualize,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()
