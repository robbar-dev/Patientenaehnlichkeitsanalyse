import pandas as pd

# Define the paths to the input and output files
input_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\V2\nlst_subset_v2.csv"
output_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\V2\analysis_output.txt"

# Read the input CSV file
df = pd.read_csv(input_file_path, engine='python')

# Group by the columns '51', '59', '61' and count the number of occurrences for each combination
combination_counts = df.groupby(['51', '59', '61']).size().reset_index(name='count')

# Prepare the text content
output_lines = ["51|59|61|count|"]
for _, row in combination_counts.iterrows():
    output_lines.append(f"{row['51']}|{row['59']}|{row['61']}|{row['count']}|")

# Write the results to the output text file
with open(output_file_path, 'w') as output_file:
    output_file.write("\n".join(output_lines))

print(f"The analysis results have been saved to: {output_file_path}")
