import pandas as pd

# Define the paths to the input and output files
input_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\filtered_files\pid_structured_desc.csv"
output_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\filtered_files\pid_structured_desc_filtered_analysis.csv"

# Read the input CSV file
df = pd.read_csv(input_file_path, delimiter=';', engine='python')

# Drop the specified columns
columns_to_drop = ['60', '63', '64', '57', '56']
df = df.drop(columns=columns_to_drop)

# Define the columns that should be combined into the new LKM category
lkm_columns = ['51', '52', '53', '62']

# Create a new LKM column that is 1 if any of the lkm_columns are 1
df['LKM'] = df[lkm_columns].max(axis=1)

# Drop the original columns that have been combined into LKM
df = df.drop(columns=lkm_columns)

# Get the columns representing the binary conditions in the new structure
condition_columns = ['LKM', '65', '59', '61', '55', '54', '58']

# Group by the condition columns and count the occurrences
count_df = df[condition_columns].value_counts().reset_index(name='Count')

# Reorder columns to have 'Count' as the first column
count_df = count_df[['Count'] + condition_columns]

# Save the results to a new CSV file
count_df.to_csv(output_file_path, index=False, sep=';')
