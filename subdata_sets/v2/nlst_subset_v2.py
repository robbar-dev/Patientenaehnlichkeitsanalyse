import pandas as pd

input_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V2\nlst_subset_v2_unausgeglichen.csv"
output_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V2\nlst_subset_v2.csv"

df = pd.read_csv(input_file_path)

max_samples = 265

grouped = df.groupby(['51', '59', '61'])

balanced_df = grouped.apply(lambda x: x.sample(n=min(len(x), max_samples), random_state=42)).reset_index(drop=True)

balanced_df.to_csv(output_file_path, index=False)

print(f"Saved: {output_file_path}")
