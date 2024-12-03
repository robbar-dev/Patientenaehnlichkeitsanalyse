import pandas as pd

# Define the paths to the input and output files
input_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\nlst_subset_v1.csv"
output_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\nlst_subset_v1_analysis.csv"

# Read the input CSV file
df = pd.read_csv(input_file_path, delimiter=';', engine='python')

# Select only the relevant columns for analysis
analysis_columns = ['LKM', '59', '61']
analysis_df = df[analysis_columns]

# Count the occurrences of each unique combination
combination_counts = analysis_df.value_counts().reset_index()
combination_counts.columns = ['LKM', '59', '61', 'Count']

# Reorder columns to have 'Count' as the first column
combination_counts = combination_counts[['Count', 'LKM', '59', '61']]

# Save the results to a new CSV file
combination_counts.to_csv(output_file_path, index=False, sep=';')