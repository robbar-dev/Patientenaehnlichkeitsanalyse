import pandas as pd

# Define the paths to the input and output files
input_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\V1\nlst_subset_v1.csv"
output_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\V2\nlst_subset_v2_unausgeglichen.csv"

# Read the input CSV file with automatic delimiter detection
with open(input_file_path, 'r') as file:
    first_line = file.readline()
    delimiter = ',' if ',' in first_line else ';'

# Load the CSV file using the detected delimiter
df = pd.read_csv(input_file_path, delimiter=delimiter, engine='python')

# Filter rows where '52', '53', and '62' are all 0
filtered_df = df[(df['52'] == 0) & (df['53'] == 0) & (df['62'] == 0)]

# Select relevant columns for the output CSV
output_columns = ['pid', 'study_yr', '51', '59', '61', '52', '53', '62', 'combination', 'patient_key']
output_df = filtered_df[output_columns]

# Save the results to a new CSV file
output_df.to_csv(output_file_path, index=False, sep=delimiter)

print(f"The filtered dataset has been saved to: {output_file_path}")

