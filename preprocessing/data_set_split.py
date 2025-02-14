import pandas as pd
import os

def split_csv_balanced(
    input_csv,
    output_train_csv,
    output_val_csv,
    output_test_csv,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
):
    """
    Liest input_csv ein, gruppiert nach 'combination',
    splittet jede Gruppe in train/val/test nach den angegebenen Anteilen,
    und schreibt die Ergebnisse in output_train_csv, output_val_csv, output_test_csv.

    Args:
      input_csv: Pfad zur Original-CSV
      output_train_csv: Pfad zum Training-CSV
      output_val_csv: Pfad zum Validierungs-CSV
      output_test_csv: Pfad zum Test-CSV
      train_ratio, val_ratio, test_ratio: float, Summe ~ 1.0
      random_state: für reproducible Shuffle
    """

    df = pd.read_csv(input_csv)
    # Erwartet Spalten: 'pid', 'study_yr', 'combination'

    # Check summation
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Train/Val/Test Ration sum != 1.0 ({total_ratio})")

    df_train_list = []
    df_val_list = []
    df_test_list = []

    group_col = 'combination'
    grouped = df.groupby(group_col)

    # Schleife über jede Kategorie
    for combo, group_df in grouped:
        # Shuffle die Zeilen dieser Gruppe
        group_df = group_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        n = len(group_df)
        n_train = int(round(n * train_ratio))
        n_val   = int(round(n * val_ratio))
        # rest geht in test
        n_test  = n - n_train - n_val

        # Indizes für Split
        train_df = group_df.iloc[:n_train]
        val_df   = group_df.iloc[n_train:n_train + n_val]
        test_df  = group_df.iloc[n_train + n_val:]

        # an die globalen Listen anhängen
        df_train_list.append(train_df)
        df_val_list.append(val_df)
        df_test_list.append(test_df)

    df_train = pd.concat(df_train_list, ignore_index=True)
    df_val   = pd.concat(df_val_list,   ignore_index=True)
    df_test  = pd.concat(df_test_list,  ignore_index=True)

    df_train.to_csv(output_train_csv, index=False)
    df_val.to_csv(output_val_csv,       index=False)
    df_test.to_csv(output_test_csv,     index=False)

    print(f"Ergebnis:\n"
          f"  Train: {len(df_train)} Zeilen\n"
          f"  Val:   {len(df_val)} Zeilen\n"
          f"  Test:  {len(df_test)} Zeilen")

if __name__ == "__main__":
    input_csv = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\head_vs_lung\head_vs_lung_dataset.csv"
    out_train = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\head_vs_lung\training\nlst_subset_v5_head_vs_lung_training.csv"
    out_val   = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\head_vs_lung\val\nlst_subset_v5_head_vs_lung_val.csv"
    out_test  = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\head_vs_lung\test\nlst_subset_v5_head_vs_lung_test.csv"

    split_csv_balanced(
        input_csv=input_csv,
        output_train_csv=out_train,
        output_val_csv=out_val,
        output_test_csv=out_test,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
