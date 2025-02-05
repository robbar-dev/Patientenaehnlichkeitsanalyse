import os
import pandas as pd

# Pfade
dataPath = r"D:\thesis_robert\NLST_subset_v7_dicom_normal_unverarbeitet"
csvOutputPath = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V7\nlst_subset_v7_normal_data.csv"

# Liste für die CSV-Daten
data_list = []

# Durch alle Ordner im dataPath iterieren
for folder_name in os.listdir(dataPath):
    if not os.path.isdir(os.path.join(dataPath, folder_name)):
        continue  # Überspringe Dateien

    # Ordnername muss im Format "pid_216164_study_yr_2" sein
    parts = folder_name.split("_")
    if len(parts) == 5 and parts[0] == "pid" and parts[2] == "study" and parts[3] == "yr":
        try:
            pid = int(parts[1])  # Patient ID extrahieren
            study_yr = int(parts[4])  # Study Year extrahieren
            combination = "0-0-1"  # Feste Kombination für alle
            data_list.append([pid, study_yr, combination])
        except ValueError:
            print(f"Warnung: Konnte keine gültigen Werte aus {folder_name} extrahieren.")

# DataFrame erstellen
df = pd.DataFrame(data_list, columns=["pid", "study_yr", "combination"])

# CSV speichern
df.to_csv(csvOutputPath, index=False)

print(f"CSV-Datei gespeichert unter: {csvOutputPath}")
