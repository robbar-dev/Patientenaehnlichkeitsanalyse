import pandas as pd

# Paths to the input files
canc_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\structured_nlst_canc.csv"
ctab_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\structured_nlst_ctab.csv"

# Read the canc and ctab CSV files with error handling for malformed lines
try:
    canc_df = pd.read_csv(canc_file_path, delimiter=';', engine='python', on_bad_lines='skip')
    ctab_df = pd.read_csv(ctab_file_path, delimiter=';', engine='python', on_bad_lines='skip')
except Exception as e:
    print(f"Error reading CSV files: {e}")
    exit()

# Filter rows in ctab DataFrame where pid is in canc DataFrame
filtered_ctab_df = ctab_df[ctab_df['pid'].isin(canc_df['pid'])]

# Define the path for the output CSV file
output_filtered_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\filtered_ctab_with_canc_pids.csv"

# Save the filtered DataFrame to a new CSV file
filtered_ctab_df.to_csv(output_filtered_file_path, index=False, sep=';')

print(f"The filtered ctab data has been saved to: {output_filtered_file_path}")
