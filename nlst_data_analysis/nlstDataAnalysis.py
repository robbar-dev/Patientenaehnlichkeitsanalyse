import os
import pandas as pd
import tkinter as tk
from tkinter import scrolledtext

# Funktion, die die CSV-Dateien analysiert
def analyse_csv_files(folder_path):
    # Liste der Dateien im Ordner abrufen
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    analysis_results = []

    for csv_file in csv_files:
        # CSV-Datei laden
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path, header=0)

        # Anzahl der vollständigen Einträge berechnen
        total_rows_count = df.shape[0]  # Gesamtanzahl der Einträge
        complete_rows_count = df.dropna().shape[0]

        # Anzahl der unterschiedlichen Patienten (basierend auf 'pid')
        unique_patients_count = df['pid'].nunique() if 'pid' in df.columns else 'N/A'

        # Fehlende Kategorien zählen
        missing_counts = df.isna().sum()

        # Kategorien in vollständige und fehlende aufteilen
        complete_categories = [category for category, count in missing_counts.items() if count == 0]
        missing_categories = sorted([(category, count) for category, count in missing_counts.items() if count > 0], key=lambda x: x[1])

        # Analyseergebnisse sammeln
        analysis_results.append({
            'file_name': csv_file,
            'total_rows_count': total_rows_count,
            'unique_patients_count': unique_patients_count,
            'complete_rows_count': complete_rows_count,
            'complete_categories': complete_categories,
            'missing_categories': missing_categories
        })

    return analysis_results

# Erweiterte Analysefunktion (auskommentiert)
# def extended_analysis():
#     # Dateien laden
#     nlst_ctab_path = r'C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\csv_unstructured\nlst_ctab.csv'
#     nlst_screen_path = r'C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\csv_unstructured\nlst_screen.csv'

#     # Daten laden
#     df_ctab = pd.read_csv(nlst_ctab_path, header=0)
#     df_screen = pd.read_csv(nlst_screen_path, header=0)

#     # Analyse der Verteilung der Krankheiten (sct_ab_desc)
#     sct_ab_desc_counts = df_ctab['sct_ab_desc'].value_counts()

#     # Analyse der CT-Diagnosequalität (ctdxqual)
#     ctdxqual_counts = df_screen['ctdxqual'].value_counts()
#     ctdxqual_1_count = ctdxqual_counts.get(1, 0)
#     ctdxqual_2_count = ctdxqual_counts.get(2, 0)
#     ctdxqual_3_count = ctdxqual_counts.get(3, 0)
#     total_ctdxqual_count = ctdxqual_1_count + ctdxqual_2_count + ctdxqual_3_count

#     # Patienten-IDs mit ctdxqual 1 oder 2 filtern (nur eindeutige PIDs)
#     pids_ctdxqual_1 = df_screen[df_screen['ctdxqual'] == 1]['pid'].drop_duplicates()
#     pids_ctdxqual_2 = df_screen[df_screen['ctdxqual'] == 2]['pid'].drop_duplicates()

#     # Kombinierte Analyse von sct_ab_desc und ctdxqual (nur für gefilterte Patienten)
#     filtered_ctab_1 = df_ctab[df_ctab['pid'].isin(pids_ctdxqual_1)].drop_duplicates(subset=['pid', 'sct_ab_desc'])
#     filtered_ctab_2 = df_ctab[df_ctab['pid'].isin(pids_ctdxqual_2)].drop_duplicates(subset=['pid', 'sct_ab_desc'])
#     combined_ctdxqual_1 = filtered_ctab_1['sct_ab_desc'].value_counts()
#     combined_ctdxqual_2 = filtered_ctab_2['sct_ab_desc'].value_counts()
#     total_combined_ctdxqual_1 = combined_ctdxqual_1.sum()
#     total_combined_ctdxqual_2 = combined_ctdxqual_2.sum()
#     unique_patients_ctdxqual_1 = pids_ctdxqual_1.nunique()
#     unique_patients_ctdxqual_2 = pids_ctdxqual_2.nunique()

#     return {
#         'sct_ab_desc_counts': sct_ab_desc_counts,
#         'ctdxqual_1_count': ctdxqual_1_count,
#         'ctdxqual_2_count': ctdxqual_2_count,
#         'ctdxqual_3_count': ctdxqual_3_count,
#         'total_ctdxqual_count': total_ctdxqual_count,
#         'combined_ctdxqual_1': combined_ctdxqual_1,
#         'combined_ctdxqual_2': combined_ctdxqual_2,
#         'total_combined_ctdxqual_1': total_combined_ctdxqual_1,
#         'total_combined_ctdxqual_2': total_combined_ctdxqual_2,
#         'unique_patients_ctdxqual_1': unique_patients_ctdxqual_1,
#         'unique_patients_ctdxqual_2': unique_patients_ctdxqual_2
#     }

# Funktion, um die Ergebnisse in einem separaten Fenster anzuzeigen
def show_analysis_results(results):
    # Fenster erstellen
    root = tk.Tk()
    root.title("CSV Datenanalyse Ergebnisse")
    root.geometry("800x900")

    # Textbereich mit Scrollfunktion
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=50)
    text_area.pack(padx=10, pady=10)

    # Ursprüngliche Analyseergebnisse anzeigen
    for result in results:
        text_area.insert(tk.END, f"Dateiname: {result['file_name']}\n")
        text_area.insert(tk.END, f"Gesamtanzahl der Einträge: {result['total_rows_count']}\n")
        text_area.insert(tk.END, f"Anzahl der unterschiedlichen Patienten: {result['unique_patients_count']}\n")
        text_area.insert(tk.END, f"Anzahl der vollständigen Einträge: {result['complete_rows_count']}\n")
        text_area.insert(tk.END, "Vollständige Kategorien:\n")
        for category in result['complete_categories']:
            text_area.insert(tk.END, f"  - {category}\n")
        text_area.insert(tk.END, "Fehlende Kategorien (sortiert von kleinster zu größter Anzahl fehlender Werte):\n")
        for category, count in result['missing_categories']:
            text_area.insert(tk.END, f"  - {category}: {count} fehlende Werte\n")
        text_area.insert(tk.END, "\n" + "-"*60 + "\n\n")

    root.mainloop()

# Hauptfunktion
def main():
    # Ordnerpfad festlegen (hier anpassen)
    folder_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\csv_unstructured"

    # Analyse durchführen
    results = analyse_csv_files(folder_path)

    # Erweiterte Analyse (auskommentiert)
    # extended_results = extended_analysis()

    # Ergebnisse anzeigen
    show_analysis_results(results)

if __name__ == "__main__":
    main()
