import os
import pandas as pd
import glob

def delete_nifti_files(deleted_pids_csv, nifti_folder):
    df = pd.read_csv(deleted_pids_csv, sep=';', dtype={'pid': str, 'study_yr': str})
    for index, row in df.iterrows():
        pid = row['pid'].strip()
        study_yr = row['study_yr'].strip()

        pattern = f"pid_{pid}_study_yr_{study_yr}*.nii*"
        search_pattern = os.path.join(nifti_folder, pattern)

        # Alle matching Datein suchen
        matching_files = glob.glob(search_pattern)
        if matching_files:
            for filepath in matching_files:
                try:
                    os.remove(filepath)
                    print(f"Datei gelöscht: {filepath}")
                except Exception as e:
                    print(f"Fehler beim Löschen der Datei {filepath}: {e}")
        else:
            print(f"Keine Datei gefunden für pid {pid}, study_yr {study_yr}")

if __name__ == "__main__":
    deleted_pids_csv = r"D:\thesis_robert\deleted_pids_v3.csv"
    nifti_folder = r"D:\thesis_robert\NLST_subset_v3_nifti_unverarbeitet"

    delete_nifti_files(deleted_pids_csv, nifti_folder)
