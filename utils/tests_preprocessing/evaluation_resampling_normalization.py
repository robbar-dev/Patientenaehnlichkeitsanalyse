import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Laden der image_statistics.csv Datei mit spezifizierter Kodierung
image_stats = pd.read_csv(
    r"D:\thesis_robert\NLST_subset_v4_nifti_3mm_Voxel\validation_resampling_normalization\image_statistics.csv",
    encoding='latin1'
)

# Überprüfen der ersten Zeilen der Daten
print("Erste Zeilen der image_statistics.csv:")
print(image_stats.head())

# Konvertieren von Spalten zu numerischen Typen, falls erforderlich
numeric_columns = [
    'Mean_Intensity', 'Median_Intensity', 'Std_Intensity',
    'Voxel_Size_X', 'Voxel_Size_Y', 'Voxel_Size_Z',
    'Image_Shape_X', 'Image_Shape_Y', 'Image_Shape_Z',
    'Physical_Size_X', 'Physical_Size_Y', 'Physical_Size_Z'
]

# Konvertieren zu numerischen Typen
for col in numeric_columns:
    image_stats[col] = pd.to_numeric(image_stats[col], errors='coerce')

# Statistische Kennzahlen berechnen
intensity_stats = image_stats[['Mean_Intensity', 'Median_Intensity', 'Std_Intensity']].describe()
voxel_size_stats = image_stats[['Voxel_Size_X', 'Voxel_Size_Y', 'Voxel_Size_Z']].describe()
image_shape_stats = image_stats[['Image_Shape_X', 'Image_Shape_Y', 'Image_Shape_Z']].describe()
physical_size_stats = image_stats[['Physical_Size_X', 'Physical_Size_Y', 'Physical_Size_Z']].describe()

print("\nIntensitätsstatistiken:")
print(intensity_stats)

print("\nVoxelgrößen Statistik:")
print(voxel_size_stats)

print("\nBilddimensionen Statistik:")
print(image_shape_stats)

print("\nPhysische Größe Statistik:")
print(physical_size_stats)

# Verteilungen visualisieren
# 1. Histogramm der Mean_Intensity
sns.histplot(image_stats['Mean_Intensity'].dropna(), kde=True)
plt.title('Verteilung der mittleren Intensitätswerte')
plt.xlabel('Mittlere Intensität')
plt.ylabel('Anzahl')
plt.show()

# 2. Histogramm der Std_Intensity
sns.histplot(image_stats['Std_Intensity'].dropna(), kde=True)
plt.title('Verteilung der Standardabweichung der Intensitätswerte')
plt.xlabel('Standardabweichung der Intensität')
plt.ylabel('Anzahl')
plt.show()

# 3. Histogramm der Image_Shape_X
sns.histplot(image_stats['Image_Shape_X'].dropna(), kde=True)
plt.title('Verteilung der Bilddimensionen X')
plt.xlabel('Bilddimension X (Voxel)')
plt.ylabel('Anzahl')
plt.show()

# 4. Histogramm der Physical_Size_X
sns.histplot(image_stats['Physical_Size_X'].dropna(), kde=True)
plt.title('Verteilung der physischen Größe X')
plt.xlabel('Physische Größe X (mm)')
plt.ylabel('Anzahl')
plt.show()

# 5. Boxplot der Intensitätswerte
plt.figure(figsize=(10, 6))
sns.boxplot(data=image_stats[['Mean_Intensity', 'Median_Intensity', 'Std_Intensity']])
plt.title('Boxplot der Intensitätsstatistiken')
plt.ylabel('Wert')
plt.show()

# 6. Scatterplot: Bilddimensionen vs. Physische Größe
plt.figure(figsize=(10, 6))
plt.scatter(image_stats['Image_Shape_X'], image_stats['Physical_Size_X'], alpha=0.7)
plt.title('Bilddimensionen X vs. Physische Größe X')
plt.xlabel('Bilddimension X (Voxel)')
plt.ylabel('Physische Größe X (mm)')
plt.show()
