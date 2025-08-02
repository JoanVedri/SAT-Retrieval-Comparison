import os
from glob import glob
import numpy as np
import pandas as pd

# Script which filters and creates daily final MODIS data

# Filtering LST by error (< 1 K) and view angle (< 45º)

# Open LST day directory
os.chdir('D:\TFM\APPEEARS\LST\Estacions\Dia')
current_path = os.getcwd()

# Save filenames
file_list = glob(current_path + '\*.csv')

# Loop for filtering daytime LST
i = 0
for file in file_list:

    # Open LST files
    os.chdir('D:\TFM\APPEEARS\LST\Estacions\Dia')
    df = pd.read_csv(file,sep=',', encoding='latin-1').reset_index(drop=True)

    # Filter dates with LST error > 1 K (QC 65 or higher)
    df = df[df['QC_Day']<65]

    # Filter dates with view angle > 45º 
    df = df[abs(df['Day_view_ang']) <= 45]

    # Reset df index
    df = df.reset_index(drop=True)

    # Open final daytime LST directory
    os.chdir('D:\TFM\APPEEARS\LST\Estacions\criba\Dia')

    # Save station id
    name = file[34:39]

    # For stations that don't have letter in the id
    if name[-1] == '.':
        name = name[0:4]
    
    # Creating id column
    s = pd.Series([name]) 
    s = s.repeat(len(df))
    s = s.to_frame().reset_index(drop=True)

    df['INDICATIVO'] = s

    # Save data
    i = i+1
    print('Estacion : {i} dia'.format(i = i)) 
    df.to_csv(name +'.csv', index = False)


# Open LST night directory
os.chdir('D:\TFM\APPEEARS\LST\Estacions\ANit')
current_path = os.getcwd()

# Save filenames
file_list = glob(current_path + '\*.csv')

# Loop for filtering nighttime LST
i = 0
for file in file_list:

    # Open LST files
    df = pd.read_csv(file,sep=',', encoding='latin-1').reset_index(drop=True)

    # Filter dates with LST error > 1 K (QC 65 or higher)
    df = df[df['QC_Night']<65]

    # Filter dates with view angle > 45º 
    df = df[abs(df['Night_view_ang']) <= 45]

    # Reset df index
    df = df.reset_index(drop=True)

    # # Open final nighttime LST directory
    os.chdir('D:\TFM\APPEEARS\LST\Estacions\criba\ANit')

    # Save station id   
    name = file[35:40]

    # For stations that don't have letter in the id
    if name[-1] == '.':
        name = name[0:4]
    
    # Creating id column
    s = pd.Series([name]) 
    s = s.repeat(len(df))
    s = s.to_frame().reset_index(drop=True)

    df['INDICATIVO'] = s
    
    # Save data
    i = i+1
    print('Estacion : {i} nit'.format(i = i)) 
    df.to_csv(name +'.csv', index = False)

# Creating daily NDVI data

# Open station NDVI directory 
os.chdir('D:\TFM\APPEEARS\ANDVI\Estacions')
current_path = os.getcwd()

# Save namefiles
file_list = glob(current_path + '\*.csv')

# Loop to obtain daily NDVI from 16-days NDVI
i = 0
for file in file_list:

    # Open NDVI files
    os.chdir('D:\TFM\APPEEARS\ANDVI\Estacions')
    df = pd.read_csv(file,sep=',', encoding='latin-1').reset_index(drop=True)

    # Convert time to datetime type
    df['Time'] = pd.to_datetime(df['Time'])

    # Save station id
    name = file[32:37]

    # For stations that don't have letter in the idl
    if name[-1] == '.':
        name = name[0:4]
    
    # Save date and repeat it 15 time to cover the 16-day interval
    ndvi = df['NDVI']
    ndvi = ndvi.repeat(15)

    # Dataframe to save daily NDVI data
    ds = pd.DataFrame()

    # Fill the the date column obtaining daily NDVI data
    ds['Date'] = pd.date_range(start='1/1/2021',end='31/12/2022')

    # Creating and auxiliar variable to iterate
    days = ds['Date']
    
    # Final daily NDVI dataframe
    final_data = pd.DataFrame()

    # Loop to find the closest date and concatenate data
    for day in days:
        
        df_sort = df.iloc[(df['Time']-day).abs().argsort()[:1]]
        final_data = pd.concat([final_data,df_sort]).reset_index(drop=True) 
    
    # Saving final dates
    final_data['Date'] = ds['Date']

    # # Creating id column
    s = pd.Series([name]) 
    s = s.repeat(len(final_data))
    s = s.to_frame().reset_index(drop=True)

    final_data['INDICATIVO'] = s

    # Open final NDVI directory and saving files
    os.chdir('D:\TFM\APPEEARS\ANDVI\Daily_NDVI')
    
    i = i+1
    print('Estacion : {i} '.format(i = i)) 
    
    final_data.to_csv(name +'.csv', index = False)


# Filtering of albedo files, extracting only the mean albedo value for each date

# Open station albedo directory
os.chdir('D:\TFM\APPEEARS\Albedo\Estacions')
current_path = os.getcwd()

# Save filenames
file_list = glob(current_path + '\*.csv')

# Sorting files
file_list = sorted(file_list)

# Loop to filter the albedo values
i = 0
for file in file_list:
    ds = pd.DataFrame()
    # Open file
    os.chdir('D:\TFM\APPEEARS\Albedo\Estacions')
    df = pd.read_csv(file,sep=',', encoding='latin-1').reset_index(drop=True)
    
    # Saving date and mean albedo of 2x2 pixels
    ds['Date'] = df['Time']
    ds['Albedo'] = df['Mean']
    
    # Save station name
    name = file[33:38]
    
    # For stations that don't have letter in the id
    if name[-1] == '.':
        name = name[0:4]
    
    # Creating id column
    s = pd.Series([name]) 
    s = s.repeat(len(ds))
    s = s.to_frame().reset_index(drop=True)
    
    ds['INDICATIVO'] = s
    
    

    # Open filtered albedo directory
    os.chdir('D:\TFM\APPEEARS\Albedo\Albedo_final')
    
    i = i+1
    print('Estacion : {i} '.format(i = i)) 
    
    # Drop possible NaN values
    ds = ds.dropna()

    # Save file for each station
    ds.to_csv(name +'.csv', index = False)