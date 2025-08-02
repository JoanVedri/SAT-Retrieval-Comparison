import os
from glob import glob
import warnings
import xarray as xr
import numpy as np
import pandas as pd
import functions_SAT_inst as fs


# Open geodata file to obtain the stations list and their location

os.chdir('D:\TFM\Inputs')

X = pd.read_csv('geodata.csv',sep=',', encoding='latin-1').reset_index(drop=True)

stations_list = X['INDICATIVO'].to_list()
pos_lat = X.columns.get_loc('LATITUD')
pos_lon = X.columns.get_loc('LONGITUD')

# Open MODIS albedo directory
os.chdir('D:\TFM\APPEEARS\Albedo')
current_path = os.getcwd()

# Creating a file list with every file to iterate them
file_list = glob(current_path + '\*.nc')

albedo = pd.DataFrame()

# Loop to obtain the albedo of the 4 nearest pixels to the station in a 2x2 and calculate their mean value to reshape the albedo resolution from 500m to 1km
i = 0
for station in stations_list:
    
    values = X[X['INDICATIVO'] == station]

    os.chdir('D:\TFM\APPEEARS\ALBEDO')
    latitud = float(values.iloc[0,pos_lat])
    longitud = float(values.iloc[0,pos_lon])
    alb_st = pd.DataFrame()

    # Loop to open all albedo files and find the values over the station pixels
    for file in file_list:
        site_coords=[latitud,longitud]
        xr_data=xr.open_dataset(file,engine='netcdf4')
        lat=np.tile(xr_data.lat.values, (xr_data.dims['lon'],1)).transpose()
        lon=np.tile(xr_data.lon.values, (xr_data.dims['lat'],1))

        # Extracting the nearest corners of the albedo pixels to the station

        x1, x2, y1, y2 = fs.find_nearest_col_row_2x2(lat, lon, site_coords)
        datetimeindex = xr_data.indexes['time'].to_datetimeindex()
        data=[]

        # Extracting the data for each variable for each day, for each corner and merging them all
        for ind, time in enumerate(datetimeindex):
            swa1=xr_data.Albedo_WSA_shortwave[ind,x1,y1].values
            swa2=xr_data.Albedo_WSA_shortwave[ind,x1,y2].values
            swa3=xr_data.Albedo_WSA_shortwave[ind,x2,y1].values
            swa4=xr_data.Albedo_WSA_shortwave[ind,x2,y2].values
            qf1=xr_data.BRDF_Albedo_Band_Mandatory_Quality_shortwave[ind,x1,y1].values
            qf2=xr_data.BRDF_Albedo_Band_Mandatory_Quality_shortwave[ind,x1,y2].values
            qf3=xr_data.BRDF_Albedo_Band_Mandatory_Quality_shortwave[ind,x2,y1].values
            qf4=xr_data.BRDF_Albedo_Band_Mandatory_Quality_shortwave[ind,x2,y2].values

            # Mean of the 2x2 pixels to obtain the albedo a 1 km resolution
            swamean=xr_data.Albedo_WSA_shortwave[ind,x1:x2,y1:y2].values.mean()
            swasd=xr_data.Albedo_WSA_shortwave[ind,x1:x2,y1:y2].values.std()
            data.append([time,swa1, swa2, swa3, swa4, qf1, qf2, qf3, qf4, swamean, swasd])

        data_columns=['Time', 'swa1', 'swa2', 'swa3', 'swa4', 'qf1', 'qf2', 'qf3', 'qf4', 'Mean', 'SD']
        data=pd.DataFrame(data, columns=data_columns)
        alb_st = pd.concat([alb_st,data]).reset_index(drop=True)
   
    # Merging every day data in one dataframe
    i = i+1
    albedo = pd.concat([albedo,alb_st])

    # Creation of files for each station
    print('Estacion : {i} '.format(i = i))

    os.chdir('D:\TFM\APPEEARS\ALBEDO\Estacions')
    f = station + '.csv'
    albedo.to_csv(f, index=False)