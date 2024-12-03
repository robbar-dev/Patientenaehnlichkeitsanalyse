import pandas as pd
import os

# Paths to the input files
screen_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\structured_nlst_screen.csv"
ctab_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\structured_nlst_ctab.csv"

# Read the screen and ctab CSV files
screen_df = pd.read_csv(screen_file_path, delimiter=';', engine='python')
ctab_df = pd.read_csv(ctab_file_path, delimiter=';', engine='python')

# Filter the screen DataFrame for acceptable image quality (ctdxqual == 1 or 2)
filtered_screen_df = screen_df[screen_df['ctdxqual'].isin([1, 2])]

# Iterate over each row in the filtered screen DataFrame
filtered_rows = []
for _, screen_row in filtered_screen_df.iterrows():
    pid = screen_row['pid']
    study_yr = screen_row['study_yr']
    ctdxqual = screen_row['ctdxqual']

    # Find matching rows in the ctab DataFrame based on 'pid' and 'study_yr'
    matching_ctab_rows = ctab_df[(ctab_df['pid'] == pid) & (ctab_df['study_yr'] == study_yr)]
    
    # Append the matched rows with the corresponding 'ctdxqual' value
    for _, ctab_row in matching_ctab_rows.iterrows():
        combined_row = ctab_row.to_dict()
        combined_row['ctdxqual'] = ctdxqual
        filtered_rows.append(combined_row)

# Create a new DataFrame from the filtered rows
final_filtered_df = pd.DataFrame(filtered_rows)

# Define the path for the output CSV file
output_merged_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\filtered_patients.csv"

# Save the merged DataFrame to a new CSV file
final_filtered_df.to_csv(output_merged_file_path, index=False, sep=';')

print(f"The filtered patient data has been saved to: {output_merged_file_path}")
