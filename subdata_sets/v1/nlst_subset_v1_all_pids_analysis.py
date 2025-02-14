import pandas as pd

input_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\nlst_subset_v1.csv"
output_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\nlst_subset_v1_analysis.csv"

df = pd.read_csv(input_file_path, delimiter=';', engine='python')

analysis_columns = ['LKM', '59', '61']
analysis_df = df[analysis_columns]

combination_counts = analysis_df.value_counts().reset_index()
combination_counts.columns = ['LKM', '59', '61', 'Count']

combination_counts = combination_counts[['Count', 'LKM', '59', '61']]

combination_counts.to_csv(output_file_path, index=False, sep=';')