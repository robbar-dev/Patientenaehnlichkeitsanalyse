import json
import pandas as pd

json_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V2\output_file.json"
csv_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V2\nlst_subset_v2.csv"
output_json_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V2\nlst_subset_v2_with_combinations.json"

with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)


csv_data = pd.read_csv(csv_file_path)

for entry in json_data:
    pid = entry['pid']
    study_yr = entry['study_yr']
    combination = csv_data.loc[(csv_data['pid'] == pid) & (csv_data['study_yr'] == study_yr), 'combination']
    if not combination.empty:
        entry['combination'] = combination.values[0]
    else:
        entry['combination'] = None  

with open(output_json_path, 'w') as output_json_file:
    json.dump(json_data, output_json_file, indent=4)

print(f"Die aktualisierte JSON-Datei wurde gespeichert unter: {output_json_path}")
