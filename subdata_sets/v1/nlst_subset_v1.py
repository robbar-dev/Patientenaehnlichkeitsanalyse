import os
import pandas as pd
import pydicom
import numpy as np
import random
import re
import time

# Zufälliger Seed für Reproduzierbarkeit
random.seed(42)
np.random.seed(42)

# Pfade definieren
data_dir = r"M:\public_data\tcia_ml\nlst\ct"
csv_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\V1\nlst_subset_v1_all_pids.csv"
output_csv_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\V1\nlst_subset_v1.csv"

nlst_df = pd.read_csv(csv_path, sep=';')

nlst_df['combination'] = nlst_df[['LKM', '59', '61']].astype(str).agg('-'.join, axis=1)

selected_patients_df = pd.DataFrame()

series_report = []
failed_patients_list = []

patients_per_combination = 1153

print("Starte Auswahl der Patienten pro Kombination...")

selected_pids_per_combination = {}

for combination, group in nlst_df.groupby('combination'):
    print(f"\nVerarbeite Kombination {combination} mit {len(group)} Patienten.")
    group = group.copy()  # Kopie, um Originaldatei nicht zu ändern
    group['pid'] = group['pid'].astype(int)  
    group['study_yr'] = group['study_yr'].astype(int) 

    selected_patient_keys = set()
    failed_patient_keys = set()

    while len(selected_patient_keys) < patients_per_combination:
        # Wähle die verbleibenden Patienten aus, die noch nicht verarbeitet wurden
        group['patient_key'] = list(zip(group['pid'], group['study_yr']))
        remaining_group = group[~group['patient_key'].isin(selected_patient_keys.union(failed_patient_keys))]
        if remaining_group.empty:
            print(f"Nicht genügend Patienten in Kombination {combination}, um {patients_per_combination} zu erreichen.")
            break

        # Zufällige Auswahl eines Patienten
        patient = remaining_group.sample(n=1).iloc[0]
        patient_id = int(patient['pid'])
        study_year = int(patient['study_yr'])
        desired_year = {0: 1999, 1: 2000, 2: 2001}.get(study_year, -1)

        # Pfad zum Patientenordner
        patient_path = os.path.join(data_dir, str(patient_id))

        if not os.path.exists(patient_path):
            print(f"Patientenordner für PID {patient_id} nicht gefunden.")
            failed_patient_keys.add((patient_id, study_year))
            failed_patients_list.append({"pid": patient_id, "study_yr": study_year, "reason": "Patientenordner nicht gefunden", "combination": combination})
            continue

        # Durchsuchen der Study-Year-Ordner im Patientenordner
        series_paths = []
        for folder in os.listdir(patient_path):
            folder_path = os.path.join(patient_path, folder)
            if not os.path.isdir(folder_path):
                continue

            # Extrahieren des Jahres aus dem Ordnername
            match = re.match(r"\d{2}-\d{2}-(\d{4})", folder)
            if match:
                year = int(match.group(1))
                if year == desired_year:
                    study_path = folder_path
                    # Überprüfen, ob es Unterordner (Serien) gibt
                    if os.path.isdir(study_path):
                        for series_folder in os.listdir(study_path):
                            series_path = os.path.join(study_path, series_folder)
                            if os.path.isdir(series_path):
                                series_paths.append(series_path)
                    else:
                        series_paths.append(study_path)

        if not series_paths:
            print(f"Keine Serien für PID {patient_id}, Studienjahr {desired_year} gefunden.")
            failed_patient_keys.add((patient_id, study_year))
            failed_patients_list.append({"pid": patient_id, "study_yr": study_year, "reason": f"Keine Serien für Studienjahr {desired_year} gefunden", "combination": combination})
            continue

        # Überprüfe die Serien für Qualitätskriterien
        best_series = None
        best_metadata = {}
        for series_path in series_paths:
            try:
                dicom_files = []
                for root, _, files in os.walk(series_path):
                    dicom_files.extend([os.path.join(root, f) for f in files if f.endswith(".dcm")])

                num_slices = len(dicom_files)
                if num_slices == 0:
                    continue

                dicom = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
                slice_thickness = getattr(dicom, 'SliceThickness', None)
                pixel_spacing = getattr(dicom, 'PixelSpacing', None)

                criteria_met = True

                # Slice Thickness zwischen 1.0 mm und 2.5 mm
                if slice_thickness is None or not (1.0 <= float(slice_thickness) <= 2.5):
                    criteria_met = False

                # Pixel Spacing zwischen 0.5 mm und 1.5 mm
                if pixel_spacing is None or not all(0.5 <= float(x) <= 1.5 for x in pixel_spacing):
                    criteria_met = False

                # Anzahl der Slices über 80
                if num_slices < 80:
                    criteria_met = False

                if criteria_met:
                    best_series = series_path
                    best_metadata = {
                        "slice_thickness": slice_thickness,
                        "num_slices": num_slices,
                        "pixel_spacing": pixel_spacing
                    }
                    break 

            except Exception as e:
                print(f"Fehler beim Verarbeiten der Serie {series_path}: {e}")
                continue

        if best_series:
            # Füge den Patienten zu den ausgewählten Patienten hinzu
            selected_patient_keys.add((patient_id, study_year))
            selected_patients_df = pd.concat([selected_patients_df, patient.to_frame().T], ignore_index=True)
            series_report.append({
                "pid": patient_id,
                "study_yr": study_year,
                "series_path": best_series,
                "metadata": best_metadata,
                "combination": combination
            })
            print(f"Patient {patient_id}, Study Year {study_year} hinzugefügt. Anzahl ausgewählter Patienten für Kombination {combination}: {len(selected_patient_keys)}")
        else:
            failed_patient_keys.add((patient_id, study_year))
            failed_patients_list.append({"pid": patient_id, "study_yr": study_year, "reason": "Keine geeignete Serie gefunden", "combination": combination})
            print(f"Patient {patient_id}, Study Year {study_year} erfüllt nicht die Qualitätskriterien.")

    selected_pids_per_combination[combination] = selected_patient_keys

print("\nVerarbeitung abgeschlossen.")

print("\nErstelle Berichte...")
report_df = pd.DataFrame(failed_patients_list)
report_df.to_csv(r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\V1\failed_patients_report.csv", index=False)

series_report_df = pd.DataFrame(series_report)
series_report_df.to_csv(r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\DataSetV1\V1\series_selection_report.csv", index=False)


passed_df = selected_patients_df
passed_df.to_csv(output_csv_path, index=False)
print(f"Das gefilterte Subset wurde in {output_csv_path} gespeichert.")
print(f"Anzahl der Patienten nach Qualitätskontrolle: {len(passed_df)}")

# Überprüfen, ob 800 Patienten pro Kombination ausgewählt wurden
for combination in nlst_df['combination'].unique():
    count = len(passed_df[passed_df['combination'] == combination])
    print(f"Kombination {combination}: {count} Patienten ausgewählt.")
