import os
from glob import glob
import warnings
import cfgrib
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Open geodata file to obtain the stations list and their location
os.chdir('D:\TFM\Inputs')

X = pd.read_csv('geodata.csv',sep=',', encoding='latin-1').reset_index(drop=True)

stations_list = X['INDICATIVO'].to_list()

# Open ERA5 data directory

os.chdir('D:\TFM\era5')
current_path = os.getcwd()
file_list = glob(current_path + '\*.grib')

pos_lat = X.columns.get_loc('LATITUD')
pos_lon = X.columns.get_loc('LONGITUD')

era5 = pd.DataFrame()
i = 0

# Sort the stations 
file_list = sorted(file_list)

# Loop for obtaining ERA5 data for each station
for station in stations_list:
    
    val_stations = pd.DataFrame()

    for file in file_list:

        os.chdir('D:\TFM\era5')

        # Open grib ERA5 files
        ds = cfgrib.open_dataset(file)
        df = ds.to_dataframe()
        df=df.reset_index(level=['latitude', 'longitude'])
  
        os.chdir('D:\TFM\era5\Estacions')
        values = X[X['INDICATIVO'] == station]
        

        lat = float(values.iloc[0,pos_lat])
        lon = float(values.iloc[0,pos_lon])
        
        # Find the nearest pixels to station location considering ERA5 resolution of 0.1º
        df2=df.loc[(df['latitude'] >= lat-0.05) & (df['latitude'] <= lat+ 0.05) & (df['longitude'] >= lon- 0.05)&(df['longitude'] <= lon+ 0.05)].reset_index(drop=True)
        df2 = df2.dropna()

        # For pixels near the coast it is possible to find no data values due to sea proximity, range is wider in this case and the closest pixel is chosed
        if len(df2) == 0:
   
            mod = []
            df2=df.loc[(df['latitude'] >= lat-0.25) & (df['latitude'] <= lat+ 0.25) & (df['longitude'] >= lon- 0.25)&(df['longitude'] <= lon+ 0.25)].reset_index(drop=True)
            df2 = df2.dropna()
    
            df2['modulo'] = np.sqrt(df2['latitude']*df2['latitude'] + df2['longitude']*df2['longitude'])
            
            mod = df2['modulo'].unique()
            val_mod_min = np.abs(np.sqrt(lon*lon + lat*lat) - mod).argmin()
    
            df2 = df2[df2['modulo'] == mod[val_mod_min]]
            df2 = df2.drop(['modulo'],axis=1)
            
        # Merging all files data for each station
        val_stations = pd.concat([val_stations , df2])

    f = station + '.csv'
    i = i+1

    # Saving ERA5 data per station
    print('Estacion: {i}'.format(i = i)) 
    val_stations = val_stations.reset_index(drop=True)
    val_stations.to_csv(f)
