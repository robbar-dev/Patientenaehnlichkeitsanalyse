import pandas as pd
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Eingabe- und Ausgabe-Pfade
input_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V6\VorbereitungV6\canc_pids_downloaded.csv"
output_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V6\VorbereitungV6\canc_300_400.csv"

# CSV-Datei einlesen
logging.info("Lese die CSV-Datei ein...")
df = pd.read_csv(input_path)

# Filter auf die gewünschten de_stage Werte
logging.info("Filtere die Daten nach de_stage 310, 320, 400...")
filtered_df = df[df["de_stag"].isin([310, 320, 400])]

# Relevante Spalten auswählen
logging.info("Reduziere die Daten auf relevante Spalten...")
filtered_df = filtered_df[["pid", "study_yr", "de_stag", "lesionsize", "lc_topog", "stage_only"]]

# Speichern der neuen CSV-Datei
logging.info(f"Speichere die gefilterte CSV-Datei nach {output_path}...")
filtered_df.to_csv(output_path, index=False)

logging.info("CSV-Datei erfolgreich erstellt.")