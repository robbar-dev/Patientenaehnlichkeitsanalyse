import os
import gzip
import argparse
import logging

"""

Erstellt eine Liste von Serien, die nicht geladen werden können und folgich das resampling_normalization.py Skirpt abstürzen lassen. 

Dieses Skript ist notwendig, da das resampling_normalization.py Skript die Serien in Batches lädt und eine einfache Try-Except Implementierung nicht möglich ist. 

"""

def test_gzip_integrity(input_dir, output_file):
    # Logging einrichten
    logging.basicConfig(
        filename="gzip_integrity_check.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Liste aller .nii.gz-Dateien erstellen
    nii_gz_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".nii.gz")
    ]

    if not nii_gz_files:
        print(f"Keine .nii.gz-Dateien im Verzeichnis {input_dir} gefunden.")
        return

    corrupted_files = []

    for idx, file_path in enumerate(nii_gz_files):
        filename = os.path.basename(file_path)
        print(f"Teste Datei {idx+1}/{len(nii_gz_files)}: {filename}")
        try:
            with gzip.open(file_path, 'rb') as f:
                while True:
                    buf = f.read(1024 * 1024)
                    if not buf:
                        break
            print(f"Datei {filename} ist in Ordnung.")
            logging.info(f"Datei {filename} ist in Ordnung.")
        except EOFError as e:
            print(f"EOFError bei Datei {filename}: {e}")
            logging.error(f"EOFError bei Datei {filename}: {e}")
            corrupted_files.append(filename)
        except Exception as e:
            print(f"Fehler beim Lesen der Datei {filename}: {e}")
            logging.error(f"Fehler beim Lesen der Datei {filename}: {e}")
            corrupted_files.append(filename)

    # Zusammenfassung speichern
    if corrupted_files:
        with open(output_file, 'w') as f:
            f.write("Dateien mit GZip-Integritätsfehlern:\n")
            for fname in corrupted_files:
                f.write(f"{fname}\n")
        print(f"Liste der problematischen Dateien gespeichert unter {output_file}")
    else:
        print("Alle Dateien haben die GZip-Integritätsprüfung bestanden.")

def main():
    parser = argparse.ArgumentParser(description="Überprüfung der GZip-Integrität von NIfTI-Dateien")
    parser.add_argument("--input_dir", required=True, help="Pfad zu den .nii.gz-Dateien")
    parser.add_argument("--output_file", default="corrupted_gzip_files.txt", help="Pfad zur Ausgabedatei mit problematischen Dateien")
    args = parser.parse_args()

    test_gzip_integrity(input_dir=args.input_dir, output_file=args.output_file)

if __name__ == "__main__":
    main()

# python utils\tests_preprocessing\debug_gzipFiles_resampling_normalization.py --input_dir "D:\thesis_robert\subset_v5_seg" --output_file "D:\thesis_robert\corrupted_pids"