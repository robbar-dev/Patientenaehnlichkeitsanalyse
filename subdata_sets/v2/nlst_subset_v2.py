import pandas as pd

# Define the paths to the input and output files
input_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V2\nlst_subset_v2_unausgeglichen.csv"
output_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V2\nlst_subset_v2.csv"

# Read the input CSV file
df = pd.read_csv(input_file_path)

# Define the number of samples per combination
max_samples = 265

# Group by the columns '51', '59', and '61'
grouped = df.groupby(['51', '59', '61'])

# Sample up to 'max_samples' from each group
balanced_df = grouped.apply(lambda x: x.sample(n=min(len(x), max_samples), random_state=42)).reset_index(drop=True)

# Save the balanced dataset to a new CSV file
balanced_df.to_csv(output_file_path, index=False)

print(f"Balanced dataset saved to: {output_file_path}")
