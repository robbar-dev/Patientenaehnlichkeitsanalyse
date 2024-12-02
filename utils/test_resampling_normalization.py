import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import csv

def analyze_images(input_dir, output_dir):
    # Listen der Dateien in den Eingabe- und Ausgabeordnern abrufen
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    output_files = [f for f in os.listdir(output_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    # Sortieren der Dateilisten, um sicherzustellen, dass sie übereinstimmen
    input_files.sort()
    output_files.sort()
    
    # Überprüfen, ob die Anzahl der Dateien gleich ist
    if len(input_files) != len(output_files):
        print(f"Warnung: Anzahl der Eingabedateien ({len(input_files)}) und Ausgabedateien ({len(output_files)}) stimmen nicht überein.")
    
    # Verzeichnis zum Speichern der Histogramme erstellen
    histogram_dir = os.path.join(output_dir, "histograms")
    if not os.path.exists(histogram_dir):
        os.makedirs(histogram_dir)
    
    # CSV-Datei zum Speichern der Statistiken öffnen
    stats_file = os.path.join(output_dir, "image_statistics.csv")
    metadata_comparison_file = os.path.join(output_dir, "metadata_comparison.csv")
    
    with open(stats_file, mode='w', newline='') as stats_csv, open(metadata_comparison_file, mode='w', newline='') as metadata_csv:
        stats_fieldnames = ['Image', 'Type', 'Mean_Intensity', 'Median_Intensity', 'Std_Intensity', 
                            'Voxel_Size_X', 'Voxel_Size_Y', 'Voxel_Size_Z',
                            'Image_Shape_X', 'Image_Shape_Y', 'Image_Shape_Z',
                            'Physical_Size_X', 'Physical_Size_Y', 'Physical_Size_Z',
                            'Orientation']
        metadata_fieldnames = ['Image', 'Parameter', 'Input_Value', 'Output_Value', 'Difference']
        
        stats_writer = csv.DictWriter(stats_csv, fieldnames=stats_fieldnames)
        metadata_writer = csv.DictWriter(metadata_csv, fieldnames=metadata_fieldnames)
        
        stats_writer.writeheader()
        metadata_writer.writeheader()
        
        for input_file, output_file in zip(input_files, output_files):
            print(f"\nVerarbeite {input_file} und {output_file}")
            
            # Bilder laden
            input_img = nib.load(os.path.join(input_dir, input_file))
            output_img = nib.load(os.path.join(output_dir, output_file))
            
            # Datenarrays abrufen
            input_data = input_img.get_fdata()
            output_data = output_img.get_fdata()
            
            # Metadaten überprüfen
            print("\n--- Überprüfung der Bildmetadaten ---")
            
            # Affine-Matrizen
            input_affine = input_img.affine
            output_affine = output_img.affine
            print("Affine-Matrix des Eingabebildes:")
            print(input_affine)
            print("Affine-Matrix des Ausgabebildes:")
            print(output_affine)
            
            # Voxelgrößen
            input_voxel_sizes = input_img.header.get_zooms()
            output_voxel_sizes = output_img.header.get_zooms()
            print(f"Voxelgrößen des Eingabebildes: {input_voxel_sizes}")
            print(f"Voxelgrößen des Ausgabebildes: {output_voxel_sizes}")
            
            # Achscodes
            input_axcodes = nib.aff2axcodes(input_affine)
            output_axcodes = nib.aff2axcodes(output_affine)
            print(f"Orientierung des Eingabebildes (Achscodes): {input_axcodes}")
            print(f"Orientierung des Ausgabebildes (Achscodes): {output_axcodes}")
            
            # Bilddimensionen
            input_shape = input_data.shape
            output_shape = output_data.shape
            print(f"Dimensionen des Eingabebildes: {input_shape}")
            print(f"Dimensionen des Ausgabebildes: {output_shape}")
            
            # Physische Dimensionen
            input_physical_size = np.array(input_shape) * np.array(input_voxel_sizes)
            output_physical_size = np.array(output_shape) * np.array(output_voxel_sizes)
            print(f"Physische Größe des Eingabebildes (mm): {input_physical_size}")
            print(f"Physische Größe des Ausgabebildes (mm): {output_physical_size}")
            
            # Vergleichsergebnisse in der CSV-Datei speichern
            metadata_writer.writerow({'Image': input_file, 'Parameter': 'Affine-Matrix', 
                                      'Input_Value': input_affine.tolist(), 
                                      'Output_Value': output_affine.tolist(), 
                                      'Difference': 'N/A'})
            metadata_writer.writerow({'Image': input_file, 'Parameter': 'Voxelgrößen', 
                                      'Input_Value': input_voxel_sizes, 
                                      'Output_Value': output_voxel_sizes, 
                                      'Difference': np.array(output_voxel_sizes) - np.array(input_voxel_sizes)})
            metadata_writer.writerow({'Image': input_file, 'Parameter': 'Achscodes', 
                                      'Input_Value': ''.join(input_axcodes), 
                                      'Output_Value': ''.join(output_axcodes), 
                                      'Difference': 'N/A'})
            metadata_writer.writerow({'Image': input_file, 'Parameter': 'Bilddimensionen', 
                                      'Input_Value': input_shape, 
                                      'Output_Value': output_shape, 
                                      'Difference': np.array(output_shape) - np.array(input_shape)})
            metadata_writer.writerow({'Image': input_file, 'Parameter': 'Physische Größen', 
                                      'Input_Value': input_physical_size, 
                                      'Output_Value': output_physical_size, 
                                      'Difference': output_physical_size - input_physical_size})
            
            # Statistische Analyse der Intensitätswerte
            print("\n--- Statistische Analyse der Intensitätswerte ---")
            
            # Statistiken des Eingabebildes
            input_mean = np.mean(input_data)
            input_median = np.median(input_data)
            input_std = np.std(input_data)
            print(f"Mittelwert des Eingabebildes: {input_mean}")
            print(f"Median des Eingabebildes: {input_median}")
            print(f"Standardabweichung des Eingabebildes: {input_std}")
            
            # Statistiken des Ausgabebildes
            output_mean = np.mean(output_data)
            output_median = np.median(output_data)
            output_std = np.std(output_data)
            print(f"Mittelwert des Ausgabebildes: {output_mean}")
            print(f"Median des Ausgabebildes: {output_median}")
            print(f"Standardabweichung des Ausgabebildes: {output_std}")
            
            # Statistiken in CSV speichern
            stats_writer.writerow({
                'Image': input_file,
                'Type': 'Input',
                'Mean_Intensity': input_mean,
                'Median_Intensity': input_median,
                'Std_Intensity': input_std,
                'Voxel_Size_X': input_voxel_sizes[0],
                'Voxel_Size_Y': input_voxel_sizes[1],
                'Voxel_Size_Z': input_voxel_sizes[2],
                'Image_Shape_X': input_shape[0],
                'Image_Shape_Y': input_shape[1],
                'Image_Shape_Z': input_shape[2],
                'Physical_Size_X': input_physical_size[0],
                'Physical_Size_Y': input_physical_size[1],
                'Physical_Size_Z': input_physical_size[2],
                'Orientation': ''.join(input_axcodes)
            })
            
            stats_writer.writerow({
                'Image': output_file,
                'Type': 'Output',
                'Mean_Intensity': output_mean,
                'Median_Intensity': output_median,
                'Std_Intensity': output_std,
                'Voxel_Size_X': output_voxel_sizes[0],
                'Voxel_Size_Y': output_voxel_sizes[1],
                'Voxel_Size_Z': output_voxel_sizes[2],
                'Image_Shape_X': output_shape[0],
                'Image_Shape_Y': output_shape[1],
                'Image_Shape_Z': output_shape[2],
                'Physical_Size_X': output_physical_size[0],
                'Physical_Size_Y': output_physical_size[1],
                'Physical_Size_Z': output_physical_size[2],
                'Orientation': ''.join(output_axcodes)
            })
            
            # Histogramme erstellen und speichern
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.hist(input_data.flatten(), bins=100)
            plt.title(f'Intensitätshistogramm Originalbild: {input_file}')
            plt.xlabel('Intensität')
            plt.ylabel('Häufigkeit')
            
            plt.subplot(1, 2, 2)
            plt.hist(output_data.flatten(), bins=100)
            plt.title(f'Intensitätshistogramm Verarbeitetes Bild: {output_file}')
            plt.xlabel('Intensität')
            plt.ylabel('Häufigkeit')
            
            plt.tight_layout()
            histogram_path = os.path.join(histogram_dir, f"{os.path.splitext(input_file)[0]}_histograms.png")
            plt.savefig(histogram_path)
            plt.close()
            print(f"Histogramme gespeichert unter {histogram_path}")
    
    print("\nAnalyse abgeschlossen.")
    print(f"Statistiken gespeichert in {stats_file}")
    print(f"Metadatenvergleich gespeichert in {metadata_comparison_file}")

def main():
    input_dir = r"D:\thesis_robert\test_data"
    output_dir = r"D:\thesis_robert\test_data\resampled_normalized"
    
    analyze_images(input_dir, output_dir)

if __name__ == "__main__":
    main()
