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
    Spacingd,
    ScaleIntensityRanged,
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
    target_spacing=(1.0, 1.0, 1.0),
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
        output_postfix="resampled_normalized",
        output_ext=".nii.gz",
        output_dtype=np.float32,
        resample=False,
        channel_dim=0,
        separate_folder=False,
        print_log=True
    )

    # Transformationspipeline definieren
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=target_spacing, mode=interpolation),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000, a_max=400,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        EnsureTyped(keys=["image"])
    ])

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

    dataset = Dataset(data=nifti_files, transform=transforms)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=pad_list_data_collate
    )

    for batch_data in tqdm(loader, total=len(nifti_files), desc="Progress:"):
        try:
            batch_size_actual = batch_data["image"].shape[0]
            for i in range(batch_size_actual):
                image = batch_data["image"][i]
                meta = batch_data.get("image_meta_dict", {})
                
                if meta and "filename_or_obj" in meta:
                    filename = os.path.basename(meta["filename_or_obj"][i])
                else:
                    filename = os.path.basename(nifti_files[i]["image"])

                # Debugging-Ausgabe
                print(f"Verarbeite Datei: {filename}")
                print(f"Bildform: {image.shape}")

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
        except Exception as e:
            logging.exception(f"Fehler bei der Verarbeitung eines Batches: {e}")
            print(f"Fehler bei der Verarbeitung: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Resampling und Normalisierung von CT-Bildern")
    parser.add_argument("--input_dir", required=True, help="Pfad zu den Eingabedaten")
    parser.add_argument("--output_dir", required=True, help="Pfad zum Speichern der Ausgabedaten")
    parser.add_argument("--target_spacing", nargs=3, type=float, default=(1.0, 1.0, 1.0), help="Zielauflösung in mm")
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
