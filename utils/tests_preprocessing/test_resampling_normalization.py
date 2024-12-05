import os
import random
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import csv

def analyze_images(input_dir, output_dir, sample_size):
    # Alle Dateien im Eingabeordner abrufen
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    # Stichprobe zufällig auswählen
    sample_files = random.sample(all_files, sample_size)

    # Verzeichnisse für Ergebnisse erstellen
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Verzeichnis für Histogramme erstellen
    histogram_dir = os.path.join(output_dir, "histograms")
    if not os.path.exists(histogram_dir):
        os.makedirs(histogram_dir)
    
    # CSV-Dateien zum Speichern der Ergebnisse öffnen
    stats_file = os.path.join(output_dir, "image_statistics.csv")
    metadata_comparison_file = os.path.join(output_dir, "metadata_comparison.csv")
    
    with open(stats_file, mode='w', newline='') as stats_csv, open(metadata_comparison_file, mode='w', newline='') as metadata_csv:
        stats_fieldnames = ['Image', 'Mean_Intensity', 'Median_Intensity', 'Std_Intensity', 
                            'Voxel_Size_X', 'Voxel_Size_Y', 'Voxel_Size_Z',
                            'Image_Shape_X', 'Image_Shape_Y', 'Image_Shape_Z',
                            'Physical_Size_X', 'Physical_Size_Y', 'Physical_Size_Z',
                            'Orientation']
        metadata_fieldnames = ['Image', 'Parameter', 'Value']
        
        stats_writer = csv.DictWriter(stats_csv, fieldnames=stats_fieldnames)
        metadata_writer = csv.DictWriter(metadata_csv, fieldnames=metadata_fieldnames)
        
        stats_writer.writeheader()
        metadata_writer.writeheader()
        
        for file in sample_files:
            print(f"\nVerarbeite {file}")
            
            # Bild laden
            img = nib.load(os.path.join(input_dir, file))
            
            # Datenarrays abrufen
            data = img.get_fdata()
            
            # Metadaten überprüfen
            print("\n--- Überprüfung der Bildmetadaten ---")
            
            # Affine-Matrix
            affine = img.affine
            print("Affine-Matrix:")
            print(affine)
            
            # Voxelgrößen
            voxel_sizes = img.header.get_zooms()
            print(f"Voxelgrößen: {voxel_sizes}")
            
            # Achscodes
            axcodes = nib.aff2axcodes(affine)
            print(f"Orientierung (Achscodes): {axcodes}")
            
            # Bilddimensionen
            shape = data.shape
            print(f"Dimensionen: {shape}")
            
            # Physische Dimensionen
            physical_size = np.array(shape) * np.array(voxel_sizes)
            print(f"Physische Größe (mm): {physical_size}")
            
            # Vergleichsergebnisse in der CSV-Datei speichern
            metadata_writer.writerow({'Image': file, 'Parameter': 'Affine-Matrix', 
                                      'Value': affine.tolist()})
            metadata_writer.writerow({'Image': file, 'Parameter': 'Voxelgrößen', 
                                      'Value': voxel_sizes})
            metadata_writer.writerow({'Image': file, 'Parameter': 'Achscodes', 
                                      'Value': ''.join(axcodes)})
            metadata_writer.writerow({'Image': file, 'Parameter': 'Bilddimensionen', 
                                      'Value': shape})
            metadata_writer.writerow({'Image': file, 'Parameter': 'Physische Größe', 
                                      'Value': physical_size.tolist()})
            
            # Statistische Analyse der Intensitätswerte
            print("\n--- Statistische Analyse der Intensitätswerte ---")
            
            mean_intensity = np.mean(data)
            median_intensity = np.median(data)
            std_intensity = np.std(data)
            print(f"Mittelwert: {mean_intensity}")
            print(f"Median: {median_intensity}")
            print(f"Standardabweichung: {std_intensity}")
            
            # Statistiken in CSV speichern
            stats_writer.writerow({
                'Image': file,
                'Mean_Intensity': mean_intensity,
                'Median_Intensity': median_intensity,
                'Std_Intensity': std_intensity,
                'Voxel_Size_X': voxel_sizes[0],
                'Voxel_Size_Y': voxel_sizes[1],
                'Voxel_Size_Z': voxel_sizes[2],
                'Image_Shape_X': shape[0],
                'Image_Shape_Y': shape[1],
                'Image_Shape_Z': shape[2],
                'Physical_Size_X': physical_size[0],
                'Physical_Size_Y': physical_size[1],
                'Physical_Size_Z': physical_size[2],
                'Orientation': ''.join(axcodes)
            })
            
            # Histogramm erstellen und speichern
            plt.figure(figsize=(6, 4))
            plt.hist(data.flatten(), bins=100, color='blue', alpha=0.7)
            plt.title(f'Intensitätshistogramm: {file}')
            plt.xlabel('Intensität')
            plt.ylabel('Häufigkeit')
            
            histogram_path = os.path.join(histogram_dir, f"{os.path.splitext(file)[0]}_histogram.png")
            plt.savefig(histogram_path)
            plt.close()
            print(f"Histogram gespeichert unter {histogram_path}")
    
    print("\nAnalyse abgeschlossen.")
    print(f"Statistiken gespeichert in {stats_file}")
    print(f"Metadatenvergleich gespeichert in {metadata_comparison_file}")

def main():
    input_dir = r"D:\thesis_robert\NLST_subset_v3_nifti_resampled_normalized"
    output_dir = r"D:\thesis_robert\NLST_subset_v3_nifti_resampled_normalized\validation_resampling_normalization"
    sample_size = 50  # Stichprobengröße

    analyze_images(input_dir, output_dir, sample_size)

if __name__ == "__main__":
    main()
