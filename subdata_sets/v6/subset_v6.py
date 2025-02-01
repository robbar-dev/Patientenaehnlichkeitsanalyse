import pandas as pd
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Eingabe- und Ausgabe-Pfade
sick_lung_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V6\VorbereitungV6\subset_v6_sick_lung.csv"
healthy_lung_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V6\VorbereitungV6\subset_v6_healthy_lung.csv"
output_csv = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V6\VorbereitungV6\subset_v6.csv"

# CSV-Dateien einlesen
logging.info("Lese die CSV-Dateien ein...")
df_sick = pd.read_csv(sick_lung_path)
df_healthy = pd.read_csv(healthy_lung_path)

# Relevante Spalten auswählen und 'combination' hinzufügen
logging.info("Erstelle 'combination' für beide Datensätze...")
df_sick = df_sick[["pid", "study_yr"]]
df_sick["combination"] = "1-0-0"

df_healthy = df_healthy[["pid", "study_yr"]]
df_healthy["combination"] = "0-0-1"

# Datensätze zusammenführen
logging.info("Mergen der beiden Datensätze...")
df_combined = pd.concat([df_sick, df_healthy], ignore_index=True)

# Ergebnis speichern
logging.info(f"Speichere die zusammengeführte CSV-Datei nach {output_csv}...")
df_combined.to_csv(output_csv, index=False)

logging.info("CSV-Datei erfolgreich erstellt.")
