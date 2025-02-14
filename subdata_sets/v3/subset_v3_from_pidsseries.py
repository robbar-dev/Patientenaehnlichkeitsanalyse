import os
import pandas as pd

image_folder_path = "D:\\thesis_robert\\NLST_subset_v3_nifti_resampled_normalized"
input_csv_path = "C:\\Users\\rbarbir\\OneDrive - Brainlab AG\\Dipl_Arbeit\\Datensätze\\Subsets\\V2\\nlst_subset_v2.csv"
output_csv_path = "C:\\Users\\rbarbir\\OneDrive - Brainlab AG\\Dipl_Arbeit\\Datensätze\\Subsets\\V3\\nlst_subset_v3.csv"

image_files = os.listdir(image_folder_path)
remaining_pids = []

for file_name in image_files:
    if file_name.endswith(".nii.gz"):
        pid_study_yr = file_name.split(".nii.gz")[0].replace("pid_", "").replace("_study_yr_", ",")
        remaining_pids.append(pid_study_yr)

original_df = pd.read_csv(input_csv_path, delimiter=',')
original_df['pid_study_yr'] = original_df['pid'].astype(str) + "," + original_df['study_yr'].astype(str)

filtered_df = original_df[original_df['pid_study_yr'].isin(remaining_pids)]

filtered_df = filtered_df[['pid', 'study_yr', 'combination']]
filtered_df.to_csv(output_csv_path, index=False, sep=',')

print(f"Gefilterte CSV-Datei wurde gespeichert: {output_csv_path}")
