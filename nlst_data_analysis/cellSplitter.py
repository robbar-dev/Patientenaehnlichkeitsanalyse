import pandas as pd
import os

# Define the folder containing the CSV files
folder_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28"

# Iterate through all CSV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        # Define the path to the original CSV file
        input_file_path = os.path.join(folder_path, file_name)

        # Define the path for the new output CSV file
        output_file_path = os.path.join(folder_path, f"structured_{file_name}")

        # Read the CSV file using the proper delimiter and strip any extra quotes or whitespace
        df = pd.read_csv(input_file_path, delimiter=',', engine='python', quotechar='"')

        # Save the DataFrame to a new CSV file with semicolon as the delimiter to help Excel interpret columns properly
        df.to_csv(output_file_path, index=False, sep=';')

        print(f"The file {file_name} has been restructured and saved to: {output_file_path}")
