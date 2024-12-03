import pandas as pd
import tkinter as tk
from tkinter import scrolledtext
import tkinter.font as tkFont


def analyse_filtered_patients(file_path):
    # CSV-Datei laden
    df = pd.read_csv(file_path, header=0, delimiter=';')

    # 1. Gesamte Anzahl an Einträgen
    total_entries = df.shape[0]

    # 2. Anzahl an unterschiedlichen Patienten (basierend auf 'pid')
    unique_patients_count = df['pid'].nunique()

    # 3. Verteilung von sct_ab_desc, aufgeteilt nach ctdxqual == 1 und ctdxqual == 2
    sct_ab_desc_distribution = df['sct_ab_desc'].value_counts().to_frame(name='gesamte Einträge')
    sct_ab_desc_distribution['Diagnostisches CT (1)'] = df[df['ctdxqual'] == 1]['sct_ab_desc'].value_counts()
    sct_ab_desc_distribution['Eingeschränktes CT (2)'] = df[df['ctdxqual'] == 2]['sct_ab_desc'].value_counts()
    sct_ab_desc_distribution = sct_ab_desc_distribution.fillna(0).astype(int)

    # Summenzeile hinzufügen
    sum_row = sct_ab_desc_distribution.sum().to_frame().T
    sum_row.index = ['Summe']
    sct_ab_desc_distribution = pd.concat([sct_ab_desc_distribution, sum_row])

    # Ergebnisse in einem separaten Fenster anzeigen
    root = tk.Tk()
    root.title("Analyse der CSV-Daten")
    root.geometry("800x600")

    # Textbereich mit Scrollfunktion
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=30)
    text_area.pack(padx=10, pady=10)

    # Schriftart für den Textbereich
    font = tkFont.Font(family="Courier", size=10)
    text_area.configure(font=font)

    # Ergebnisse ausgeben
    text_area.insert(tk.END, f"Gesamte Anzahl an Einträgen: {total_entries}\n")
    text_area.insert(tk.END, f"Anzahl an unterschiedlichen Patienten (pid): {unique_patients_count}\n\n")
    text_area.insert(tk.END, f"Verteilung von sct_ab_desc:\n")
    text_area.insert(tk.END, f"{'sct_ab_desc':<15} {'gesamte Einträge':<20} {'Diagnostisches CT (1)':<25} {'Eingeschränktes CT (2)'}\n")
    text_area.insert(tk.END, "-" * 80 + "\n")
    for index, row in sct_ab_desc_distribution.iterrows():
        text_area.insert(tk.END, f"{index:<15} {row['gesamte Einträge']:<20} {row['Diagnostisches CT (1)']:<25} {row['Eingeschränktes CT (2)']}\n")

    root.mainloop()


# Pfad zur CSV-Datei festlegen
file_path = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\NLST\package-nlst-780.2021-05-28\filtered_files\filtered_pid_after_removal.csv"

# Analyse durchführen
analyse_filtered_patients(file_path)
