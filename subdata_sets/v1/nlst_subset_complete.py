import pandas as pd

input_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\filtered_files\pid_structured_desc.csv"
output_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\nlst_subset_complete.csv"

df = pd.read_csv(input_file_path, delimiter=';', engine='python')

# Neue Kathegorien definieren
lkm_columns = ['51', '52', '53', '62']
nth_columns = ['55', '54', '58']
exc_columns = ['65', '60', '63', '64', '56', '57']

df['LKM'] = df[lkm_columns].max(axis=1)
df['nth'] = df[nth_columns].max(axis=1)
df['exc'] = df[exc_columns].max(axis=1)

columns_order = ['pid', 'study_yr', 'LKM', '59', '61', 'nth', '55', '54', '58', 'exc', '65', '60', '63', '64', '56', '57', '51', '52', '53', '62']
remaining_columns = [col for col in df.columns if col not in columns_order]
final_columns_order = columns_order + remaining_columns

output_df = df[final_columns_order]

output_df.to_csv(output_file_path, index=False, sep=';')
