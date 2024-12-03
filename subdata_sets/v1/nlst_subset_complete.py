import pandas as pd

# Define the paths to the input and output files
input_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\filtered_files\pid_structured_desc.csv"
output_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\nlst_subset_complete.csv"

# Read the input CSV file
df = pd.read_csv(input_file_path, delimiter=';', engine='python')

# Define the columns that should be combined into new categories
lkm_columns = ['51', '52', '53', '62']
nth_columns = ['55', '54', '58']
exc_columns = ['65', '60', '63', '64', '56', '57']

# Create new columns that combine the specified categories
df['LKM'] = df[lkm_columns].max(axis=1)
df['nth'] = df[nth_columns].max(axis=1)
df['exc'] = df[exc_columns].max(axis=1)

# Reorganize the columns in the desired order
columns_order = ['pid', 'study_yr', 'LKM', '59', '61', 'nth', '55', '54', '58', 'exc', '65', '60', '63', '64', '56', '57', '51', '52', '53', '62']
remaining_columns = [col for col in df.columns if col not in columns_order]
final_columns_order = columns_order + remaining_columns

# Create the output DataFrame with the desired structure
output_df = df[final_columns_order]

# Save the results to a new CSV file
output_df.to_csv(output_file_path, index=False, sep=';')
