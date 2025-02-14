import os
import pandas as pd
import pydicom
import re
import json


csv_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V2\nlst_subset_v2_ausgeglichen_ohnePaths.csv"  
data_root = r"M:\public_data\tcia_ml\nlst\ct"  
output_csv_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V2\nlst_subset_v2.csv"  
log_file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V2\fehlerlogs.txt"  
output_json_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V2\output_file.json"  

def prioritize_series(series_metadata, single_folder):
    """Priorisiert Serien basierend auf den Auswahlkriterien."""
    priorities = {"STANDARD": 1, "LUNG": 2, "BONE": 3, "OTHER": 4, "onlyOne": 5}
    
    if single_folder and not series_metadata:
        return {
            "priority": "onlyOne",
            "SliceThickness": float('inf'),
            "PixelSpacing": None,
            "dicom_path": single_folder,
            "SeriesDescription": "Only one folder available"
        }
    
    # Sortiere nach Priorität, SliceThickness und PixelSpacing
    sorted_series = sorted(
        series_metadata,
        key=lambda x: (
            priorities.get(x['priority'], 4),  # Priorität
            x['SliceThickness'],  # Kleinere SliceThickness bevorzugt
            min(x['PixelSpacing']) if x['PixelSpacing'] else float('inf')  # Feineres PixelSpacing bevorzugt
        )
    )
    return sorted_series[0]  

def find_series_with_priority(data_root, csv_df, log_file_path):
    mappings = []
    metadata_list = []
    errors = []

    total_patients = len(csv_df)
    for idx, row in csv_df.iterrows():
        pid = int(row['pid'])
        study_yr = int(row['study_yr'])
        desired_year = {0: 1999, 1: 2000, 2: 2001}.get(study_yr, -1)

        print(f"[{idx+1}/{total_patients}] Verarbeite PID {pid}, Studienjahr {study_yr}")
        
        patient_path = os.path.join(data_root, str(pid))
        if not os.path.exists(patient_path):
            errors.append(f"Patientenordner für PID {pid} nicht gefunden.")
            continue

        series_metadata = []
        single_folder_path = None 
        study_path = None

        for folder in os.listdir(patient_path):
            folder_path = os.path.join(patient_path, folder)
            if not os.path.isdir(folder_path):
                continue

            # Jahr aus dem Ordnernamen extrahieren
            match = re.match(r"\d{2}-\d{2}-(\d{4})", folder)
            if match:
                year = int(match.group(1))
                if year == desired_year:
                    study_path = folder_path
                    subfolders = [os.path.join(study_path, sf) for sf in os.listdir(study_path) if os.path.isdir(os.path.join(study_path, sf))]
                    
                    # Speichere den einzigen Unterordner, falls vorhanden
                    if len(subfolders) == 1:
                        single_folder_path = subfolders[0]

                    for series_folder in subfolders:
                        dicom_files = [f for f in os.listdir(series_folder) if f.endswith('.dcm')]
                        if not dicom_files:
                            continue

                        try:
                            first_dicom = os.path.join(series_folder, dicom_files[0])
                            ds = pydicom.dcmread(first_dicom)
                            
                            # Bestimme Priorität basierend auf SeriesDescription
                            series_description = getattr(ds, 'SeriesDescription', '').upper()
                            if "STANDARD" in series_description:
                                priority = "STANDARD"
                            elif "LUNG" in series_description:
                                priority = "LUNG"
                            elif "BONE" in series_description:
                                priority = "BONE"
                            else:
                                priority = "OTHER"

                            # Metadaten sammeln
                            series_metadata.append({
                                "pid": pid,
                                "study_yr": study_yr,
                                "SeriesDescription": series_description,
                                "SliceThickness": getattr(ds, 'SliceThickness', float('inf')),
                                "PixelSpacing": getattr(ds, 'PixelSpacing', None),
                                "priority": priority,
                                "dicom_path": series_folder
                            })

                        except Exception as e:
                            errors.append(f"Fehler: DICOM-Datei konnte nicht gelesen werden: {series_folder}, Fehler: {e}")

        # Wähle die Serie mit der höchsten Priorität oder kennzeichne als onlyOne
        if study_path:
            selected_series = prioritize_series(series_metadata, single_folder_path)
            mappings.append({
                "pid": pid,
                "study_yr": study_yr,
                "combination": row['combination'],
                "dicom_path": selected_series["dicom_path"],
                "selection_criteria": selected_series["priority"]
            })
            metadata_list.append(selected_series)
        else:
            errors.append(f"Warnung: Keine passende Serie gefunden für PID {pid}, Studienjahr {desired_year}.")

    if errors:
        with open(log_file_path, 'w') as log_file:
            log_file.write("\n".join(errors))

    return mappings, metadata_list

def sanitize_metadata(metadata_list):
    """Konvertiert komplexe Typen in JSON-kompatible Formate."""
    sanitized_list = []
    for item in metadata_list:
        sanitized_item = {}
        for key, value in item.items():
            if isinstance(value, (list, tuple)):
                sanitized_item[key] = list(value)  
            elif isinstance(value, (int, float, str)) or value is None:
                sanitized_item[key] = value  # Belasse kompatible Typen
            else:
                sanitized_item[key] = str(value)  # Andere Typen zu String
        sanitized_list.append(sanitized_item)
    return sanitized_list


csv_df = pd.read_csv(csv_path)
mappings, metadata_list = find_series_with_priority(data_root, csv_df, log_file_path)
sanitized_metadata_list = sanitize_metadata(metadata_list)

# Ergebnis speichern
pd.DataFrame(mappings).to_csv(output_csv_path, index=False)
with open(output_json_path, 'w') as json_file:
    json.dump(sanitized_metadata_list, json_file, indent=4)

print(f"Mapping gespeichert unter: {output_csv_path}")
print(f"Metadaten gespeichert unter: {output_json_path}")
if os.path.exists(log_file_path):
    print(f"Fehler-Log erstellt unter: {log_file_path}")
else:
    print("Keine Fehler aufgetreten.")
