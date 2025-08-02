import os
import numpy as np
import pandas as pd
import functions_SAT_inst as fs




# Main Script for Instantaneous SAT retrieval using linear and nonlinear methods. 
# Linear methods: OLS, Ridge, Lasso, Elastic-Net. 
# Nonlinear methods: Random forest, XGBoost, KNN and MLP neuronal network 


# Extraction of geographical data of AEMET stations from DEM. Slope, aspect calculated on QGIS. Mean altitud calculaded on SNAP. Other data are extracted from AEMET files #

# AEMET data directory
os.chdir('D:\TFM\Dades') 

# Open AEMET master files for each basin 
df1 = pd.read_csv('MaestroCuenca_0_2021_2022.csv',sep=';', encoding='latin-1').reset_index(drop=True)
df2 = pd.read_csv('MaestroCuenca_6_2021_2022.csv',sep=';', encoding='latin-1').reset_index(drop=True)
df3 = pd.read_csv('MaestroCuenca_7_2021_2022.csv',sep=';', encoding='latin-1').reset_index(drop=True)
df4 = pd.read_csv('MaestroCuenca_8_2021_2022.csv',sep=';', encoding='latin-1').reset_index(drop=True)
df5 = pd.read_csv('MaestroCuenca_9_2021_2022.csv',sep=';', encoding='latin-1').reset_index(drop=True)
df6 = pd.read_csv('MaestroCuenca_B_2021_2022.csv',sep=';', encoding='latin-1').reset_index(drop=True)

# Creating one database with all basins
maestro = pd.concat([df1, df2, df3, df4, df5, df6])
maestro = maestro.reset_index(drop=True)

# Geographical data directory
os.chdir('D:\TFM\DEM')  

# Open geographical data files 
dem = pd.read_csv('slope_aspect.csv',sep=',', encoding='latin-1').reset_index(drop=True)
mean_altitude = pd.read_csv('dades_qgis.csv',sep=',', encoding='latin-1').reset_index(drop=True)
altitude = pd.read_csv('altitudes.csv',sep=';', encoding='latin-1').reset_index(drop=True)

# Preparing variables to calculate mean difference of altitude.
m = maestro['ALTITUD'].astype(float).reset_index(drop=True)
mean = mean_altitude['AHmean'].reset_index(drop=True)
h = altitude['band_1'].reset_index(drop=True)

# Fill dataframe with stations id, name and geographical data 
df = pd.DataFrame()

df['INDICATIVO'] = maestro['INDICATIVO']
df['NOMBRE'] = maestro['NOMBRE']
df['LATITUD'] = maestro['LATITUD'].str.replace(',','.')
df['LONGITUD'] = maestro['LONGITUD'].str.replace(',','.')
df['ALTITUD'] = maestro['ALTITUD']
df['ALTITUD_DEM'] = altitude['band_1']
df['AH_AEMET'] = m - mean
df['AH_DEM'] = h - mean
df['SLOPE'] = dem['Slope1']
df['ASPECT'] = dem['Aspect1']

# Empty array to calculate distance to coast of each station
a = np.empty(len(df))

# Converting stations to iterable list
stations_list = df['INDICATIVO'].to_list()

# Counter starts
i = 0

# Distance to coast calculation using coast function described in functions_SAT_inst file
for station in stations_list:
    values = df[df['INDICATIVO'] == station]
    lat = float(values.iloc[0,2])
    lon = float(values.iloc[0,3])
    x = fs.coast(lat,lon)
    a[i] = x 
    i = i + 1
    print(i)

df1 = pd.DataFrame(a)
df['COAST'] = df1.reset_index(drop=True)
df = df.reset_index(drop=True)

# Save geograghical variables for each station
df.to_csv('D:\TFM\Inputs\geodata.csv')  



