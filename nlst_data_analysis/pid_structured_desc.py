import pandas as pd

# Define the path to the input file
input_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\filtered_files\filtered_pid_after_removal.csv"

# Read the input CSV file
df = pd.read_csv(input_file_path, delimiter=';', engine='python')

# Get unique pid and study_yr combinations
unique_combinations = df[['pid', 'study_yr']].drop_duplicates()

# Create a list of all unique sct_ab_desc values, sorted from 51 to 65
unique_conditions = sorted([int(desc) for desc in df['sct_ab_desc'].unique() if desc.isdigit() and 51 <= int(desc) <= 65])

# Initialize an empty list to store the rows for the new dataframe
rows = []

# Iterate through each unique pid and study_yr combination
for _, row in unique_combinations.iterrows():
    pid = row['pid']
    study_yr = row['study_yr']
    
    # Create a dictionary to store the new row's data
    new_row = {'pid': pid, 'study_yr': study_yr}
    
    # Add columns for each condition and set them to 0 initially
    for condition in unique_conditions:
        new_row[str(condition)] = 0
    
    # Filter the original dataframe to get the rows corresponding to the current pid and study_yr
    filtered_rows = df[(df['pid'] == pid) & (df['study_yr'] == study_yr)]
    
    # Set the condition columns to 1 if the condition is present for the current pid and study_yr
    for condition in filtered_rows['sct_ab_desc']:
        if condition.isdigit() and 51 <= int(condition) <= 65:
            new_row[str(condition)] = 1
    
    # Add the remaining columns from the original dataframe (excluding pid, study_yr, and sct_ab_desc)
    remaining_columns = filtered_rows.iloc[0].drop(labels=['pid', 'study_yr', 'sct_ab_desc']).to_dict()
    new_row.update(remaining_columns)
    
    # Append the new row to the list of rows
    rows.append(new_row)

# Create a new dataframe from the list of rows
new_df = pd.DataFrame(rows)

# Define the path for the output CSV file
output_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\filtered_files\pid_structured_desc.csv"

# Save the new dataframe to a CSV file
new_df.to_csv(output_file_path, index=False, sep=';')

print(f"The restructured patient data has been saved to: {output_file_path}")

