
import os
import sys
import pandas as pd

def main():
    """
    Dieses Skript filtert Deine große CSV (mit 7 Kombinationen)
    auf nur 2 Klassen: '0-0-1' vs '0-1-0'.

    Resultat: '2class_subset.csv' mit diesen beiden Klassen.
    """
    # 1) Pfade anpassen
    original_csv = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\nlst_subset_v5.csv"
    output_csv   = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\classification_test\nlst_subset_v5_2classes.csv"

    # 2) Lies original CSV
    df = pd.read_csv(original_csv)

    # 3) Filter auf die 2 Klassen
    target_classes = ['0-0-1', '0-1-0']
    df_2classes = df[df['combination'].isin(target_classes)]

    # 4) Falls Balancierung, checke Class-Count    
    # => optionaler code
    class_counts = df_2classes['combination'].value_counts()
    print(class_counts)

    # 5) Speichere Subset
    df_2classes.to_csv(output_csv, index=False)
    print(f"Gespeichert: {output_csv} mit {len(df_2classes)} Zeilen.")

if __name__ == "__main__":
    main()
