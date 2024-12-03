import os
import pydicom

def analyze_dicom_metadata(folder_path):
    # Liste aller DICOM-Dateien im Ordner
    dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]
    if not dicom_files:
        print(f"Keine DICOM-Dateien in {folder_path} gefunden.")
        return
    
    # Eine DICOM-Datei laden und Metadaten anzeigen
    dicom_file = dicom_files[0]
    ds = pydicom.dcmread(dicom_file)
    
    # Relevante Metadaten
    print("SeriesDescription:", getattr(ds, 'SeriesDescription', 'Nicht verfügbar'))
    print("Modality:", getattr(ds, 'Modality', 'Nicht verfügbar'))
    print("SliceThickness:", getattr(ds, 'SliceThickness', 'Nicht verfügbar'))
    print("PixelSpacing:", getattr(ds, 'PixelSpacing', 'Nicht verfügbar'))
    print("ImageType:", getattr(ds, 'ImageType', 'Nicht verfügbar'))

# Beispielaufruf
analyze_dicom_metadata(r"M:\public_data\tcia_ml\nlst\ct\111110\01-02-1999-NA-NLST-LSS-42011\2.000000-0OPAGELSQXB3502.512048.00.01.5-36617")
analyze_dicom_metadata(r"M:\public_data\tcia_ml\nlst\ct\111110\01-02-1999-NA-NLST-LSS-42011\3.000000-0OPAGELSQXD3502.512048.00.01.5-37196")
analyze_dicom_metadata(r"M:\public_data\tcia_ml\nlst\ct\127020\01-02-1999-NA-NLST-LSS-96277\2.000000-0OPAGEHSQXD3402.512056.00.11.5-03758")
analyze_dicom_metadata(r"M:\public_data\tcia_ml\nlst\ct\127020\01-02-1999-NA-NLST-LSS-96277\3.000000-0OPAGEHSQXB3402.512056.00.11.5-74502")
analyze_dicom_metadata(r"M:\public_data\tcia_ml\nlst\ct\122196\01-02-2001-NA-NLST-LSS-81470\2.000000-2OPAGEHSQXD2802.512056.00.11.5-04434")
analyze_dicom_metadata(r"M:\public_data\tcia_ml\nlst\ct\122196\01-02-2001-NA-NLST-LSS-81470\3.000000-2OPAGEHSQXB2802.512056.00.11.5-49396")
analyze_dicom_metadata(r"M:\public_data\tcia_ml\nlst\ct\103857\01-02-2001-NA-NLST-LSS-83178\3.000000-2OPATOAQUL4FC51270.32-66946")
