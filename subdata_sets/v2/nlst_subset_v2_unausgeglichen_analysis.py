import pandas as pd

input_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\V2\nlst_subset_v2.csv"
output_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\V2\analysis_output.txt"

df = pd.read_csv(input_file_path, engine='python')

combination_counts = df.groupby(['51', '59', '61']).size().reset_index(name='count')

output_lines = ["51|59|61|count|"]
for _, row in combination_counts.iterrows():
    output_lines.append(f"{row['51']}|{row['59']}|{row['61']}|{row['count']}|")

with open(output_file_path, 'w') as output_file:
    output_file.write("\n".join(output_lines))

print(f"Saved: {output_file_path}")
