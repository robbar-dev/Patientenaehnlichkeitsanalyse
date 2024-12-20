import pandas as pd

def calculate_slice_statistics(csv_path):
    try:
        # CSV-Datei einlesen
        df = pd.read_csv(csv_path, encoding='latin1')

        # Filtere die Zeilen, die 'Bilddimensionen' im Parameter enthalten
        bilddimensionen_df = df[df['Parameter'] == 'Bilddimensionen']

        # Extrahiere die Slice-Anzahl (dritter Wert im 'Value'-Tuple)
        bilddimensionen_df['Slice_Count'] = bilddimensionen_df['Value'].str.extract(r'\(\d+, \d+, (\d+)\)').astype(int)

        # Berechne die Statistiken
        average_slices = bilddimensionen_df['Slice_Count'].mean()
        max_slices = bilddimensionen_df['Slice_Count'].max()
        min_slices = bilddimensionen_df['Slice_Count'].min()

        # Ergebnisse ausgeben
        print(f"Durchschnittliche Slice-Anzahl: {average_slices:.2f}")
        print(f"Maximale Slice-Anzahl: {max_slices}")
        print(f"Minimale Slice-Anzahl: {min_slices}")

    except FileNotFoundError:
        print(f"Die Datei unter dem Pfad {csv_path} wurde nicht gefunden.")
    except Exception as e:
        print(f"Es ist ein Fehler aufgetreten: {e}")

if __name__ == "__main__":
    # Pfad zur CSV-Datei angeben
    csv_path = r"D:\thesis_robert\NLST_subset_v4_nifti_3mm_Voxel\validation_resampling_normalization\metadata_comparison.csv"

    # Funktion aufrufen
    calculate_slice_statistics(csv_path)
