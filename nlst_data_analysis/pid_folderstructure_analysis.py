import os
import pandas as pd

# Paths to the folder structure and ctab file
data_folder_path = r"M:\public_data\tcia_ml\nlst\ct"
ctab_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\structured_nlst_ctab.csv"

# Collect all PIDs from the folder structure
folder_pids = [folder_name for folder_name in os.listdir(data_folder_path) if os.path.isdir(os.path.join(data_folder_path, folder_name))]

# Read the ctab CSV file
ctab_df = pd.read_csv(ctab_file_path, delimiter=';', engine='python')
ctab_pids = ctab_df['pid'].astype(str).unique().tolist()

# Find PIDs that are in the folder structure but not in ctab
pids_in_folders_not_in_ctab = list(set(folder_pids) - set(ctab_pids))

# Find PIDs that are in ctab but not in the folder structure
pids_in_ctab_not_in_folders = list(set(ctab_pids) - set(folder_pids))

# Create a DataFrame for each list
pids_in_folders_not_in_ctab_df = pd.DataFrame(pids_in_folders_not_in_ctab, columns=['pid'])
pids_in_ctab_not_in_folders_df = pd.DataFrame(pids_in_ctab_not_in_folders, columns=['pid'])

# Define the output file path
output_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\pid_folder_vs_ctab_comparison.xlsx"

# Write the results to an Excel file with two sheets
with pd.ExcelWriter(output_file_path) as writer:
    pids_in_folders_not_in_ctab_df.to_excel(writer, sheet_name='PIDs_in_folders_not_in_ctab', index=False)
    pids_in_ctab_not_in_folders_df.to_excel(writer, sheet_name='PIDs_in_ctab_not_in_folders', index=False)

print(f"Comparison results saved to: {output_file_path}")
