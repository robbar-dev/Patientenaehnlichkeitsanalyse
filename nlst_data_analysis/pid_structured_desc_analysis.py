import pandas as pd

# Define the path to the input and output files
input_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\filtered_files\pid_structured_desc.csv"
output_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\filtered_files\pid_structured_desc_analysis.csv"

# Read the input CSV file
df = pd.read_csv(input_file_path, delimiter=';', engine='python')

# Get the columns representing the binary conditions
condition_columns = ['65', '64', '51', '52', '59', '60', '53', '56', '61', '55', '63', '58', '62', '54', '57']

# Group by the condition columns and count the occurrences
count_df = df[condition_columns].value_counts().reset_index(name='Count')

# Reorder columns to have 'Count' as the first column
count_df = count_df[['Count'] + condition_columns]

# Save the results to a new CSV file
count_df.to_csv(output_file_path, index=False, sep=';')
