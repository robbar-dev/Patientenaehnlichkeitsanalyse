import pandas as pd

# Define the paths to the input files
filtered_pid_with_ctdxqual_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\filtered_files\filtered_pid_with_ctdxqual.csv"
folder_structure_ctab_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\filtered_files\folderStructure_ctab.xlsx"

# Read the CSV and Excel files
filtered_pid_df = pd.read_csv(filtered_pid_with_ctdxqual_path, delimiter=';', engine='python')
folder_structure_ctab_df = pd.read_excel(folder_structure_ctab_path, sheet_name='PIDs_in_ctab_not_in_folders')

# Extract the list of PIDs to be removed
pids_to_remove = folder_structure_ctab_df['pid'].tolist()

# Filter the dataframe to remove rows with PIDs in the exclusion list and keep only rows with ctdxqual == 1
filtered_pid_df = filtered_pid_df[(~filtered_pid_df['pid'].isin(pids_to_remove)) & (filtered_pid_df['ctdxqual'] == 1)]

# Define the path for the output CSV file
output_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\filtered_files\filtered_pid_after_removal.csv"

# Save the filtered DataFrame to a new CSV file
filtered_pid_df.to_csv(output_file_path, index=False, sep=';')

print(f"The filtered patient data has been saved to: {output_file_path}")
