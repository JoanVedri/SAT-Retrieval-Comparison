import os
from glob import glob
import pandas as pd


# Script which merges all MODIS data deleting those days when some variable is missing

# Open daytime filtered LST files
os.chdir('D:\TFM\APPEEARS\LST\Estacions\criba\Dia')
current_path = os.getcwd()

# Save filenames
file_list_lst_dia = glob(current_path + '\*.csv')

# Open nighttime filtered LST files
os.chdir('D:\TFM\APPEEARS\LST\Estacions\criba\ANit')
current_path = os.getcwd()

# Save filenames
file_list_lst_nit = glob(current_path + '\*.csv')

# Open filtered albedo files
os.chdir('D:\TFM\APPEEARS\Albedo\Albedo_final')
current_path = os.getcwd()

# Save filenames
file_list_albedo = glob(current_path + '\*.csv')


# Open filtered NDVI files
os.chdir('D:\TFM\APPEEARS\ANDVI\Daily_NDVI')
current_path = os.getcwd()

# Save filenames
file_list_ndvi = glob(current_path + '\*.csv')

# Open geographical data file
os.chdir('D:\TFM\Inputs')
geodata = pd.read_csv('geodata.csv',sep=',', encoding='latin-1').reset_index(drop=True)


# Final daytime LST dataframe
df_final_lst_dia = pd.DataFrame()

# Merging all station data in one file
i_lst = 0
for file in file_list_lst_dia:

    df_aux = pd.DataFrame()
    df_lst = pd.read_csv(file,sep=',', encoding='latin-1').reset_index(drop=True)

    if len(df_lst) != 0:

        df_aux['INDICATIVO'] = df_lst['INDICATIVO']
        df_aux['Date'] = df_lst['Time']
        df_aux['Day_view_time'] = df_lst['Day_view_time']
        df_aux['LST_day'] = df_lst['LST_day']
        
        a = df_aux['INDICATIVO'].unique()

        s = geodata[geodata['INDICATIVO'] == a[0]]
        s = pd.concat([s]*len(df_aux)).reset_index(drop=True)
        
        df_aux ['LATITUD'] = s['LATITUD']
        df_aux ['LONGITUD'] = s['LONGITUD']
        df_aux ['ALTITUD'] = s['ALTITUD']
        df_aux ['ALTITUD_DEM'] = s['ALTITUD_DEM']
        df_aux ['AH_AEMET'] = s['AH_AEMET']
        df_aux ['AH_DEM'] = s['AH_DEM']
        df_aux ['SLOPE'] = s['SLOPE']
        df_aux ['ASPECT'] = s['ASPECT']
        df_aux ['COAST'] = s['COAST']

        df_final_lst_dia = pd.concat([df_final_lst_dia,df_aux]).reset_index(drop=True)
        i_lst = i_lst+1
        print('Estacion : {i} dia'.format(i = i_lst)) 


# Final nighttime LST dataframe
df_final_lst_nit = pd.DataFrame()

# Merging all station data in one file
i_lst = 0
for file in file_list_lst_nit:

    df_aux = pd.DataFrame()
    df_lst = pd.read_csv(file,sep=',', encoding='latin-1').reset_index(drop=True)

    if len(df_lst) != 0:

        df_aux['INDICATIVO'] = df_lst['INDICATIVO']
        df_aux['Date'] = df_lst['Time']
        df_aux['Night_view_time'] = df_lst['Night_view_time']
        df_aux['LST_night'] = df_lst['LST_night']
        
        a = df_aux['INDICATIVO'].unique()
        s = geodata[geodata['INDICATIVO'] == a[0]]
        s = pd.concat([s]*len(df_aux)).reset_index(drop=True)
        
        df_aux ['LATITUD'] = s['LATITUD']
        df_aux ['LONGITUD'] = s['LONGITUD']
        df_aux ['ALTITUD'] = s['ALTITUD']
        df_aux ['ALTITUD_DEM'] = s['ALTITUD_DEM']
        df_aux ['AH_AEMET'] = s['AH_AEMET']
        df_aux ['AH_DEM'] = s['AH_DEM']
        df_aux ['SLOPE'] = s['SLOPE']
        df_aux ['ASPECT'] = s['ASPECT']
        df_aux ['COAST'] = s['COAST']

        df_final_lst_nit = pd.concat([df_final_lst_nit,df_aux])
        i_lst = i_lst+1
        print('Estacion : {i} nit'.format(i = i_lst))  


# Final NDVI dataframe
df_final_ndvi = pd.DataFrame()

# Merging all station data in one file
i_lst = 0
for file in file_list_ndvi:

    df_aux = pd.DataFrame()
    df_ndvi = pd.read_csv(file,sep=',', encoding='latin-1').reset_index(drop=True)

    df_aux['INDICATIVO'] = df_ndvi['INDICATIVO']
    df_aux['Date'] = df_ndvi['Date']
    df_aux['NDVI'] = df_ndvi['NDVI']

    df_final_ndvi = pd.concat([df_final_ndvi,df_aux]).reset_index(drop=True)
    i_lst = i_lst+1
    print('Estacion : {i} NDVI'.format(i = i_lst)) 


# Final albedo dataframe
df_final_albedo = pd.DataFrame()

# Merging all station data in one file
i_lst = 0
for file in file_list_albedo:

    df_aux = pd.DataFrame()
    df_alb = pd.read_csv(file,sep=',', encoding='latin-1').reset_index(drop=True)

    df_aux['INDICATIVO'] = df_alb['INDICATIVO']
    df_aux['Date'] = df_alb['Date']
    df_aux['Albedo'] = df_alb['Albedo']

    df_final_albedo = pd.concat([df_final_albedo,df_aux]).reset_index(drop=True)
    i_lst = i_lst+1
    print('Estacion : {i} albedo'.format(i = i_lst)) 

# Merging all variables on date and station
df_dia = df_final_lst_dia.merge(df_final_ndvi,how= 'inner', on=['Date', 'INDICATIVO'])
df_dia = df_dia.merge(df_final_albedo,how= 'inner', on=['Date', 'INDICATIVO'])

df_nit = df_final_lst_nit.merge(df_final_ndvi,how= 'inner', on=['Date', 'INDICATIVO'])
df_nit = df_nit.merge(df_final_albedo,how= 'inner', on=['Date', 'INDICATIVO'])

# Drop NaN values 
df_dia = df_dia.dropna()
df_nit = df_nit.dropna()

# Saving daytime and nighttime files
os.chdir('D:\TFM\APPEEARS\Dades')

df_dia.to_csv('Modis_Dia.csv',index = False)
df_nit.to_csv('Modis_Nit.csv',index = False)