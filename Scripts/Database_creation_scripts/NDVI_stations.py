import os
from glob import glob
import warnings
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions_SAT_inst as fs


# Open geodata file to obtain the stations list and their location

os.chdir('D:\TFM\Inputs')

X = pd.read_csv('geodata.csv',sep=',', encoding='latin-1').reset_index(drop=True)

pos_lat = X.columns.get_loc('LATITUD')
pos_lon = X.columns.get_loc('LONGITUD')


stations_list = X['INDICATIVO'].to_list()

# Open MODIS NDVI directory
os.chdir('D:\TFM\APPEEARS\ANDVI')
current_path = os.getcwd()

# Creating a file list with every file to iterate them
file_list = glob(current_path + '\*.nc')

# Loop to extract NDVI values and quality band for each station 
i = 0
for station in stations_list:

    # Find the station and obtain only its data
    values = X[X['INDICATIVO'] == station]
    
    # Counter to find the number of stations that were used
    i = i+1
    
    # Position of station
    latitud = float(values.iloc[0,pos_lat])
    longitud = float(values.iloc[0,pos_lon])

    # Empty dataframe for NDVI
    df = pd.DataFrame()

    # Loop to open all NDVI files and find the values over the station pixel
    for file in file_list:

        site_coords=[latitud,longitud]

        # Open LST data from MYD13A2 MODIS product,
        xr_data=xr.open_dataset(file,engine='netcdf4')
    
        lat=np.tile(xr_data.lat.values, (xr_data.dims['lon'],1)).transpose()
        lon=np.tile(xr_data.lon.values, (xr_data.dims['lat'],1))
        
        # Day of the image
        datetimeindex = xr_data.indexes['time'].to_datetimeindex()
        data=[]

        # Finding the closest pixel
        x, y = fs.find_nearest_col_row(lat, lon, site_coords)

        # Extracting the data for each variable for each day and merging them all
        for ind, time in enumerate(datetimeindex):
            
            NDVI = xr_data._1_km_16_days_NDVI[ind,x,y].values
            QC = xr_data._1_km_16_days_VI_Quality[ind,x,y].values

           
            data.append([time,NDVI,QC])
        
        data_columns=['Time', 'NDVI','QC']
        data=pd.DataFrame(data, columns=data_columns)
        df = pd.concat([df,data])

    # Creation of files for each station
    print('Estacion: {i}'.format(i = i)) 
    os.chdir('D:\TFM\APPEEARS\ANDVI\Estacions')
    f = station + '.csv'
    df.to_csv(f, index=False)