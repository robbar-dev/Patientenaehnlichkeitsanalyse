import pandas as pd

input_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\V1\nlst_subset_v1.csv"
output_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\V2\nlst_subset_v2_unausgeglichen.csv"

with open(input_file_path, 'r') as file:
    first_line = file.readline()
    delimiter = ',' if ',' in first_line else ';'

df = pd.read_csv(input_file_path, delimiter=delimiter, engine='python')

filtered_df = df[(df['52'] == 0) & (df['53'] == 0) & (df['62'] == 0)]

output_columns = ['pid', 'study_yr', '51', '59', '61', '52', '53', '62', 'combination', 'patient_key']
output_df = filtered_df[output_columns]

output_df.to_csv(output_file_path, index=False, sep=delimiter)

print(f"saved: {output_file_path}")

