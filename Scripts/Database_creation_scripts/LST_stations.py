import os
from glob import glob
import warnings
import xarray as xr
import numpy as np
import pandas as pd
import functions_SAT_inst as fs
import matplotlib.pyplot as plt

# Open geodata file to obtain the stations list and their location

os.chdir('D:\TFM\Inputs')

X = pd.read_csv('geodata.csv',sep=',', encoding='latin-1').reset_index(drop=True)

pos_lat = X.columns.get_loc('LATITUD')
pos_lon = X.columns.get_loc('LONGITUD')

stations_list = X['INDICATIVO'].to_list()

# Open MODIS LST directory
os.chdir('D:\TFM\APPEEARS\LST')
current_path = os.getcwd()

# Creating a file list with every file to iterate them
file_list = glob(current_path + '\*.nc')

# Loop to extract daytime and nighttime LST values, hours of satellites surpass, view angle, quality band, and lat lon for each station 
i = 0
for station in stations_list:

    # Find the station and obtain only its data
    values = X[X['INDICATIVO'] == station]
    
    # Counter to find the number of stations that were used
    i = i+1
    
    # Position of station
    latitud = float(values.iloc[0,pos_lat])
    longitud = float(values.iloc[0,pos_lon])

    # Empty dataframes to separate daytime and nightime data
    df_day = pd.DataFrame()
    df_night = pd.DataFrame()

    # Loop to open all LST files and find the values over the station pixel
    for file in file_list:

        site_coords=[latitud,longitud]

        # Open LST data from MYD11A1 MODIS product
        xr_data=xr.open_dataset(file,engine='netcdf4')
        
        # Separate the different variables

        #Angle of view
        Day_view_ang = xr_data.variables['Day_view_angl']
        Night_view_ang = xr_data.variables['Night_view_angl']

        #Surpass hours 
        Day_view_time = xr_data.variables['Day_view_time']
        Night_view_time = xr_data.variables['Night_view_time']

        # Quality band values
        QC_Day = xr_data.variables['QC_Day']
        QC_Night = xr_data.variables['QC_Night']
        
        # LST values      
        LST_day = xr_data.variables['LST_Day_1km']
        LST_night = xr_data.variables['LST_Night_1km']

        # Geographical position
        lat=np.tile(xr_data.lat.values, (xr_data.dims['lon'],1)).transpose()
        lon=np.tile(xr_data.lon.values, (xr_data.dims['lat'],1))

        # Day of the image
        datetimeindex = xr_data.indexes['time'].to_datetimeindex()

        data_day = []
        data_night = []

        # Finding the closest pixel
        x, y = fs.find_nearest_col_row(lat, lon, site_coords)

        # Extracting the data for each variable for each day and merging them all
        for ind, time in enumerate(datetimeindex):
            Day_view_ang = xr_data.Day_view_angl[ind,x,y].values
            Night_view_ang = xr_data.Night_view_angl[ind,x,y].values
            Day_view_time = xr_data.Day_view_time[ind,x,y].values
            Night_view_time = xr_data.Night_view_time[ind,x,y].values
            QC_Day = xr_data.QC_Day[ind,x,y].values
            QC_Night=xr_data.QC_Night[ind,x,y].values
            LST_day=xr_data.LST_Day_1km[ind,x,y].values
            LST_night=xr_data.LST_Night_1km[ind,x,y].values
           
            data_day.append([time,Day_view_ang, Day_view_time, QC_Day, LST_day])
            data_night.append([time, Night_view_ang,  Night_view_time, QC_Night, LST_night])
            
        # Creating daytime and nighttime dataframes for each day
        data_columns_day=['Time', 'Day_view_ang', 'Day_view_time', 'QC_Day', 'LST_day']
        data_columns_night = ['Time', 'Night_view_ang', 'Night_view_time', 'QC_Night', 'LST_night']
        data_day=pd.DataFrame(data_day, columns=data_columns_day)
        data_night=pd.DataFrame(data_night, columns=data_columns_night)

        # Merging every day data in daytime and nighttime dataframes
        df_day = pd.concat([df_day,data_day])
        df_night = pd.concat([df_night,data_night])

    # Changing type of the variables to float and deleting those missing pixels due to errors on the adquisition
    df_day['LST_day'] = df_day['LST_day'].astype(float)
    df_day['QC_Day'] = df_day['QC_Day'].astype(float)
    df_day['Day_view_time'] = df_day['Day_view_time'].astype(float)
    df_day['Day_view_ang'] = df_day['Day_view_ang'].astype(float)
    df_day = df_day.dropna()

    df_night['LST_night'] = df_night['LST_night'].astype(float)
    df_night['QC_Night'] = df_night['QC_Night'].astype(float)
    df_night['Night_view_time'] = df_night['Night_view_time'].astype(float)
    df_night['Night_view_ang'] = df_night['Night_view_ang'].astype(float)
    df_night = df_night.dropna()

    # Creation of files for each station
    print('Estacion: {i}'.format(i = i)) 
    os.chdir('D:\TFM\APPEEARS\LST\Estacions\Dia')
    f = station + '.csv'
    df_day.to_csv(f, index=False)

    os.chdir('D:\TFM\APPEEARS\LST\Estacions\ANit')
    df_night.to_csv(f, index=False)