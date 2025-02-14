import pandas as pd

input_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\filtered_files\pid_structured_desc.csv"
output_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\nlst_subset_v1_unfiltered.csv"

df = pd.read_csv(input_file_path, delimiter=';', engine='python')

lkm_columns = ['51', '52', '53', '62']
nth_columns = ['55', '54', '58']
exc_columns = ['65', '60', '63', '64', '56', '57']

df['LKM'] = df[lkm_columns].max(axis=1)
df['nth'] = df[nth_columns].max(axis=1)
df['exc'] = df[exc_columns].max(axis=1)

filtered_df = df[(df['LKM'] == 1) | (df['59'] == 1) | (df['61'] == 1)]
filtered_df = filtered_df[(filtered_df['nth'] == 0) & (filtered_df['exc'] == 0)]

output_columns = ['pid', 'study_yr', 'LKM', '59', '61', '51', '52', '53', '62']
output_df = filtered_df[output_columns]

output_df.to_csv(output_file_path, index=False, sep=';')
