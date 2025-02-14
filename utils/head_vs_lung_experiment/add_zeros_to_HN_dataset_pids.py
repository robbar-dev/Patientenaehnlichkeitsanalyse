import csv

input_csv = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\head_vs_lung\val\nlst_subset_v5_head_vs_lung_val.csv"
output_csv = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\head_vs_lung\val\nlst_subset_v5_head_vs_lung_vals.csv"

with open(input_csv, mode='r', newline='', encoding='utf-8') as fin, \
     open(output_csv, mode='w', newline='', encoding='utf-8') as fout:
    
    reader = csv.reader(fin)
    writer = csv.writer(fout)

    header = next(reader)  
    writer.writerow(header)  
    
    for row in reader:
       
        pid = row[0]
        study_yr = row[1]
        combination = row[2]
        
        # Prüfen, ob pid rein numerisch ist
        # z.B. "103624" => isdigit() => True
        # "HNCHUM" => isdigit() => False
        if not pid.isdigit():
            # => pid enthält Buchstaben => study_yr null-auffüllen (3-stellig)
            # => "5" -> "005"
            study_yr_padded = str(int(study_yr)).zfill(3)
            row[1] = study_yr_padded
        
        writer.writerow(row)

print(f"Fertig. Neue Datei => {output_csv}")
