import pandas as pd
import json
import random
import os

input_csv_path = "C:\\Users\\rbarbir\\OneDrive - Brainlab AG\\Dipl_Arbeit\\Datensätze\\Subsets\\V2\\nlst_subset_v2.csv"
input_json_path = "C:\\Users\\rbarbir\\OneDrive - Brainlab AG\\Dipl_Arbeit\\Datensätze\\Subsets\\V2\\nlst_subset_v2.json"

output_path = "C:\\Users\\rbarbir\\OneDrive - Brainlab AG\\Dipl_Arbeit\\Datensätze\\Subsets\\V3"
os.makedirs(output_path, exist_ok=True)
output_csv_path = os.path.join(output_path, "nlst_subset_v3.csv")
output_json_path = os.path.join(output_path, "nlst_subset_v3.json")
output_deleted_pids_path = os.path.join(output_path, "deleted_pids.csv")

pids_to_remove = [
    "103168,2", "107945,0", "108213,0", "116207,1",
    "120487,0", "123521,2", "124441,0", "129949,0",
    "130047,1", "132870,1", "201886,1", "203384,2",
    "204594,1", "211129,0"
]

df = pd.read_csv(input_csv_path, delimiter=',')
df['pid_study_yr'] = df['pid'].astype(str) + "," + df['study_yr'].astype(str)
df_filtered = df[~df['pid_study_yr'].isin(pids_to_remove)]

grouped = df_filtered.groupby('combination', group_keys=False)
max_samples = 254
balanced_df = grouped.apply(lambda x: x.sample(n=min(len(x), max_samples), random_state=42)).reset_index(drop=True)

balanced_df = balanced_df.drop(columns=['pid_study_yr'])
balanced_df.to_csv(output_csv_path, index=False, sep=',')

deleted_pids = set(pids_to_remove)
for combination, group in df_filtered.groupby('combination'):
    if len(group) > max_samples:
        sampled_to_remove = group.sample(n=len(group) - max_samples, random_state=42)
        deleted_pids.update(sampled_to_remove['pid_study_yr'].tolist())

deleted_pids_df = pd.DataFrame([pid.split(',') for pid in deleted_pids], columns=['pid', 'study_yr'])
deleted_pids_df['study_yr'] = deleted_pids_df['study_yr'].astype(int)
deleted_pids_df['combination'] = deleted_pids_df.apply(lambda row: f"'{df.loc[(df['pid'] == int(row['pid'])) & (df['study_yr'] == row['study_yr']), 'combination'].values[0]}'", axis=1)
deleted_pids_df.to_csv(output_deleted_pids_path, index=False, sep=';', quoting=1, date_format='%Y-%m-%d', float_format='%.0f')

with open(input_json_path, 'r') as json_file:
    json_data = json.load(json_file)

json_filtered = [entry for entry in json_data if f"{entry['pid']},{entry['study_yr']}" not in deleted_pids]

with open(output_json_path, 'w') as json_output:
    json.dump(json_filtered, json_output, indent=4)

print(f"Balanced dataset saved to: {output_csv_path}")
print(f"Deleted PIDs saved to: {output_deleted_pids_path}")
print(f"Updated JSON file saved to: {output_json_path}")
