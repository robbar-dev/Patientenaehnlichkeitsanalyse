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

# Logging einrichten
logging.basicConfig(
    filename="resample_normalize.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def resample_and_normalize(
    input_dir,
    output_dir,
    target_spacing=(1, 1, 1),
    interpolation="trilinear",
    visualize=False,
    batch_size=4,
    num_workers=8,
    a_min=-950,  # Feinanpassung z. B. für Emphyseme
    a_max=-500,  # Bessere Sichtbarkeit für Lungenfenster
    gamma=1.5    # Kontrastanpassung
):
    """
    Führt Resampling und Normalisierung auf CT-Volumes durch.
    Überspringt bereits vorhandene Volumes (mit gleichem pid_study_yr) im output_dir.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --------------------------------------------------------------------------------
    # 1) Bereits vorhandene Dateien im Output-Ordner ermitteln (Core-Namen ohne .nii.gz)
    # --------------------------------------------------------------------------------
    existing_cores = set()
    for file in os.listdir(output_dir):
        if file.endswith(".nii.gz"):
            # Entferne das Suffix:   "pid_123_study_yr_0.nii.gz" -> "pid_123_study_yr_0"
            core_name = file.rsplit(".nii.gz", 1)[0]
            existing_cores.add(core_name)

    # --------------------------------------------------------------------------------
    # 2) Alle NIfTI-Dateien im input_dir sammeln, die NICHT bereits existieren
    # --------------------------------------------------------------------------------
    data_list = []
    all_files = [f for f in os.listdir(input_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]

    for f in all_files:
        # Core-Name bestimmen
        if f.endswith(".nii.gz"):
            core_name = f.rsplit(".nii.gz", 1)[0]
        else:  # .nii
            core_name = f.rsplit(".nii", 1)[0]

        if core_name in existing_cores:
            logging.info(f"Überspringe bereits vorhandene Datei: {f}")
            continue

        # Dateipfad ins Data-Listing aufnehmen
        data_list.append({"image": os.path.join(input_dir, f), "core_name": core_name})

    # --------------------------------------------------------------------------------
    # 3) Falls keine Dateien übrig, Abbruch
    # --------------------------------------------------------------------------------
    if not data_list:
        logging.warning(f"Keine neuen NIfTI-Dateien zu verarbeiten in {input_dir}.")
        print(f"Keine neuen NIfTI-Dateien zu verarbeiten in {input_dir}.")
        return

    # --------------------------------------------------------------------------------
    # 4) MONAI-Transforms & SaveImage
    # --------------------------------------------------------------------------------
    save_transform = SaveImage(
        output_dir=output_dir,   # Output-Verzeichnis
        output_postfix="",       # kein Extra-Suffix
        output_ext=".nii.gz",    # genau 1x .nii.gz anfügen
        resample=False,
        separate_folder=False,
        print_log=True,
    )

    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="LPS"),
        Rotate90d(keys=["image"], k=2, spatial_axes=(0, 1)),  # "Tisch unter Patient"
        Spacingd(keys=["image"], pixdim=target_spacing, mode=interpolation),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=a_min, a_max=a_max,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        AdjustContrastd(keys=["image"], gamma=gamma),
        EnsureTyped(keys=["image"], data_type="tensor")
    ])

    dataset = Dataset(data=data_list, transform=transforms)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=pad_list_data_collate
    )

    # --------------------------------------------------------------------------------
    # 5) Verarbeitung
    # --------------------------------------------------------------------------------
    total_files = len(data_list)
    processed_files_count = 0

    for batch_data in tqdm(loader, total=len(loader), desc="Progress:"):
        try:
            batch_size_actual = len(batch_data["image"])
            for i in range(batch_size_actual):
                image = batch_data["image"][i]
                meta = image.meta

                core_name = batch_data["core_name"][i]  # aus unserem data_list

                # Wir überschreiben das Meta-Feld "filename_or_obj" mit dem core_name
                # => MONAI generiert hinterher: "<output_dir>/<core_name>.nii.gz"
                meta["filename_or_obj"] = core_name

                # Debug-Ausgabe
                print(f"Verarbeite Datei: {core_name}")
                print(f"Bildform: {image.shape}")

                # Bild speichern
                save_transform(img=image, meta_data=meta)

                # Log
                logging.info(f"Datei gespeichert: {core_name}.nii.gz")
                print(f"Datei gespeichert: {os.path.join(output_dir, core_name+'.nii.gz')}")

                # Neu in existing_cores aufnehmen, um evtl. doppelte im gleichen Batch zu skippen
                existing_cores.add(core_name)

                # Optional: Visualisierung
                if visualize:
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
    parser.add_argument("--target_spacing", nargs=3, type=float, default=(1, 1, 1), help="Zielauflösung in mm")
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
        a_min=-200,   
        a_max=1200,   
        gamma=0.8      
    )

if __name__ == "__main__":
    main()



# python3.11 preprocessing\resampling_normalization.py --input_dir "D:\thesis_robert\subset_v7\NLST_subset_v7_normal_unverarbeitet" --output_dir "D:\thesis_robert\NLST_subset_v7_normal_resampled"

