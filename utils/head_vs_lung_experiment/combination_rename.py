import os
import sys
import pandas as pd

INPUT_CSV  = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\head_vs_lung\head_vs_lung_dataset.csv"
OUTPUT_CSV = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\head_vs_lung_dataset_new.csv"

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Datei {INPUT_CSV} existiert nicht.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, sep=",", engine="python")
    print(f"Einlesen OK, Anzahl Zeilen: {len(df)}")
    print(f"Spalten: {df.columns.tolist()}")

    required_cols = {"pid", "study_yr", "combination"}
    if not required_cols.issubset(df.columns):
        print("CSV muss Spalten pid, study_yr, combination enthalten.")
        sys.exit(1)

    # Kombi als string strippen
    df["combination"] = df["combination"].astype(str).str.strip()

    # Mapping-Funktion für die Kombis
    def map_combination(old_val):
        """
        HEAD => '0' -> '0-0-2'
        LUNG => '1' -> '0-0-1'
        Andere Werte => unverändert
        """
        if old_val == "0":
            return "0-0-2"
        elif old_val == "1":
            return "0-0-1"
        else:
            return old_val

    df["combination"] = df["combination"].apply(map_combination)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Fertig. Neue CSV geschrieben nach {OUTPUT_CSV}.")

    uniques = df["combination"].unique()
    print("Unique combination-Werte im neuen CSV:", uniques)

if __name__ == "__main__":
    main()

