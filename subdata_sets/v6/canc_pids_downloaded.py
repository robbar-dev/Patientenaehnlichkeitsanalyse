import os
import pandas as pd

# Pfade definieren
csv_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\csv_unstructured\nlst_canc.csv"
data_folder = r"M:\public_data\tcia_ml\nlst\ct"
output_csv_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V6\VorbereitungV6\canc_pids_downloaded.csv"

# CSV-Datei einlesen
df = pd.read_csv(csv_path, dtype=str)

# PID-Spalte bestimmen (angenommen, sie heißt 'pid')
if 'pid' not in df.columns:
    raise ValueError("Die Spalte 'pid' wurde in der CSV-Datei nicht gefunden.")

# Liste der vorhandenen PIDs im Datenordner
existing_pids = set(os.listdir(data_folder))

# Nur Zeilen behalten, deren PID im Datenordner existiert
filtered_df = df[df['pid'].isin(existing_pids)]

# Gefilterte Daten speichern
filtered_df.to_csv(output_csv_path, index=False)

print(f"Gefilterte CSV-Datei wurde gespeichert: {output_csv_path}")
