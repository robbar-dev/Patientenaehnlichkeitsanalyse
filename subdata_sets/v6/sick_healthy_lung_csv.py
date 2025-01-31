import pandas as pd
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Eingabe- und Ausgabe-Pfade
input_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V6\VorbereitungV6\pid_structured_desc.csv"
output_path_sick_lung = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V6\VorbereitungV6\subset_v6_sick_lung.csv"
output_path_healthy_lung = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V6\VorbereitungV6\subset_v6_healthy_lung.csv"

# CSV-Datei einlesen
logging.info("Lese die CSV-Datei ein...")
df = pd.read_csv(input_path, delimiter=';')
logging.debug(f"Erste Zeilen der Datei:\n{df.head()}")

# Erstellen der eindeutigen Patienten-ID
logging.info("Erstelle eindeutige Patienten-IDs (upid)...")
df["upid"] = df["pid"].astype(str) + "_" + df["study_yr"].astype(str)
logging.debug(f"Erste generierte upids:\n{df[['upid', 'study_yr']].head()}")

# Sicherstellen, dass alle relevanten Spalten vorhanden sind
relevant_columns = ["pid", "study_yr", "60", "59", "53", "62", "58", "54", "51", "56", "57", "63", "64", "52", "55", "61", "65"]
df = df[relevant_columns + ["upid"]]

# Subset 1: Sick Lung (373 upids nach den angegebenen Kriterien, mit 60 = 0)
logging.info("Filtern der kranken Lungen mit 60 = 0...")
df_sick = df[df["60"] == 0]
logging.debug(f"Anzahl gefilterter Patienten mit 60=0: {len(df_sick)}")
selected_upids = []

# Bedingung 1: 59 & 53 & 62 = 1
selected_upids.extend(df_sick[(df_sick["59"] == 1) & (df_sick["53"] == 1) & (df_sick["62"] == 1)]["upid"].tolist())
# Bedingung 2: 59 & 53 & 58 = 1
selected_upids.extend(df_sick[(df_sick["59"] == 1) & (df_sick["53"] == 1) & (df_sick["58"] == 1)]["upid"].tolist())
# Bedingung 3: 59 & 53 & 54 = 1
selected_upids.extend(df_sick[(df_sick["59"] == 1) & (df_sick["53"] == 1) & (df_sick["54"] == 1)]["upid"].tolist())

# Fehlende upids mit 59 & 53 & 51 = 1 und study_yr = 2 auffüllen
all_candidates_1 = df_sick[(df_sick["59"] == 1) & (df_sick["53"] == 1) & (df_sick["51"] == 1) & (df_sick["study_yr"] == 2)]["upid"].tolist()
for upid in all_candidates_1:
    if len(selected_upids) >= 373:
        break
    if upid not in selected_upids:
        selected_upids.append(upid)

# Falls noch Plätze frei sind, mit 59 & 53 = 1 und study_yr = 2 auffüllen
all_candidates_2 = df_sick[(df_sick["59"] == 1) & (df_sick["53"] == 1) & (df_sick["study_yr"] == 2)]["upid"].tolist()
for upid in all_candidates_2:
    if len(selected_upids) >= 373:
        break
    if upid not in selected_upids:
        selected_upids.append(upid)

logging.debug(f"Gesammelte upids für subset_v6_sick_lung: {len(selected_upids)}")

# Gefilterte Daten für subset_v6_sick_lung.csv
sick_lung_df = df_sick[df_sick["upid"].isin(selected_upids)][relevant_columns]
sick_lung_df.to_csv(output_path_sick_lung, index=False)
logging.info(f"subset_v6_sick_lung.csv gespeichert mit {len(sick_lung_df)} Einträgen.")

# Subset 2: Healthy Lung
logging.info("Filtern der gesunden Lungen...")
healthy_lung_df = df[(df["60"] == 1) & (df[["59", "53", "62", "58", "54", "51", "56", "57", "63", "64", "52", "55", "61", "65"]] == 0).all(axis=1)]
logging.debug(f"Anzahl gefilterter gesunder Patienten: {len(healthy_lung_df)}")
healthy_lung_df = healthy_lung_df[relevant_columns]
healthy_lung_df.to_csv(output_path_healthy_lung, index=False)
logging.info(f"subset_v6_healthy_lung.csv gespeichert mit {len(healthy_lung_df)} Einträgen.")

logging.info("CSV-Dateien erfolgreich erstellt.")
