import os
from glob import glob
import numpy as np
import pandas as pd
import functions_SAT_inst as fs

os.chdir('D:\TFM\APPEEARS\Dades')
dLSTd = pd.read_csv('Modis_Dia_CoordSolar.csv',sep=',', encoding='latin-1').reset_index(drop=True)
dLSTn = pd.read_csv('Modis_Nit.csv',sep=',', encoding='latin-1').reset_index(drop=True)


# Open in situ datafiles
os.chdir('D:\TFM\Dades')
df1 = pd.read_csv('Cuenca_0_2021_2022.csv',sep=';', encoding='latin-1')
df2 = pd.read_csv('Cuenca_6_2021_2022.csv',sep=';', encoding='latin-1')
df3 = pd.read_csv('Cuenca_7_2021_2022.csv',sep=';', encoding='latin-1')
df4 = pd.read_csv('Cuenca_8_2021_2022.csv',sep=';', encoding='latin-1')
df5 = pd.read_csv('Cuenca_9_2021_2022.csv',sep=';', encoding='latin-1')
df6 = pd.read_csv('Cuenca_B_2021_2022.csv',sep=';', encoding='latin-1')

# Concatenate in situ data
dades = pd.concat([df1, df2, df3, df4, df5, df6])

# Changing decimal separator
dades['TA'] = dades['TA'].str.replace(',','.')
dades['VV10M'] = dades['VV10M'].str.replace(',','.')
dades['HR'] = dades['HR'].str.replace(',','.')

# Drop NaN considering only air temperature
dades = dades.dropna(subset=['TA'])

# Creating datetime type column for in situ date data
time = pd.DataFrame()
time['year'] = dades['AÑO']
time['month'] = dades['MES']
time['day'] = dades['DIA']
time['hour'] = dades['HORA']
time['minute'] = dades['MINUTO']

date = pd.to_datetime(time)

dades['Time'] = date

# In situ database
dAEMET = dades.reset_index(drop=True)

# Open reanalysis data
os.chdir('D:\TFM\era5\Estacions')
current_path = os.getcwd()

file_list = glob(current_path + '\*.csv')
file_list = sorted(file_list)
dERA5 = pd.DataFrame()

#Loop for merging each station data in one database
i = 0
for file in file_list:
    df = pd.read_csv(file,sep=',', encoding='latin-1').reset_index(drop=True)

    name = file[22:27]
    if name[-1] == '.':
        name = name[0:4]

    s = pd.Series([name]) 
    s = s.repeat(len(df))
    s = s.to_frame().reset_index(drop=True)

    df['INDICATIVO'] = s
    
    dERA5 = pd.concat([dERA5,df])
    i = i+1
    print('Estacion : {i} '.format(i = i)) 


# Save starting date (1/1/2019) for temporal interpolation
inicio = pd.to_datetime("2019-01-01 00:00:00")

# Open geographical data and save stations list
os.chdir('D:\TFM\Inputs')

X = pd.read_csv('geodata.csv',sep=',', encoding='latin-1').reset_index(drop=True)
stations_list = X['INDICATIVO'].to_list()

# Empty dataframe for daytime and nightime databases
dAEMETd = pd.DataFrame()
dAEMETn = pd.DataFrame()
dERA5d = pd.DataFrame()
dERA5n = pd.DataFrame()
data_LSTd = pd.DataFrame()
data_LSTn = pd.DataFrame()

# Loop to drop those days without in situ, reanalysis and MODIS data
i = 0
for station in stations_list:

    # Getting data for each station
    dAEMET_aux = dAEMET[dAEMET[' INDICATIVO'] == station].reset_index(drop = True)
    dERA5_aux = dERA5[dERA5['INDICATIVO'] == station].reset_index(drop = True)
    dLSTd_aux = dLSTd[dLSTd['INDICATIVO'] == station].reset_index(drop = True)
    dLSTn_aux = dLSTn[dLSTn['INDICATIVO'] == station].reset_index(drop = True)
    
    # Coverting dates to datetime type
    diasAEMET = pd.to_datetime(dAEMET_aux['Time']).dt.date
    diasERA5 = pd.to_datetime(dERA5_aux['valid_time']).dt.date
    diasLST = pd.to_datetime(dLSTd_aux['Date']).dt.date
    nitsLST = pd.to_datetime(dLSTn_aux['Date']).dt.date
    
    # Drop LST data for those days without in situ or reanalysis data
    dLSTd_aux = dLSTd_aux[diasLST.isin(diasAEMET)] 
    dLSTn_aux = dLSTn_aux[nitsLST.isin(diasAEMET)]
    dLSTd_aux = dLSTd_aux[diasLST.isin(diasERA5)] 
    dLSTn_aux = dLSTn_aux[nitsLST.isin(diasERA5)]

    diasLST = pd.to_datetime(dLSTd_aux['Date']).dt.date
    nitsLST = pd.to_datetime(dLSTn_aux['Date']).dt.date

    # Drop in situ data for those days without LST data
    dAEMETd_aux = dAEMET_aux[diasAEMET.isin(diasLST)] 
    dAEMETn_aux = dAEMET_aux[diasAEMET.isin(nitsLST)]

    # Drop reanalysis data for those days without LST data
    dERA5d_aux = dERA5_aux[diasERA5.isin(diasLST)] 
    dERA5n_aux = dERA5_aux[diasERA5.isin(nitsLST)]

    # Drop those stations with less than 20 days for the whole period (2 years) for statistical reasons
    if len(dLSTd_aux)>20:
        dAEMETd = pd.concat([dAEMETd,dAEMETd_aux],ignore_index= True)
        dERA5d = pd.concat([dERA5d,dERA5d_aux],ignore_index= True)
        data_LSTd = pd.concat([data_LSTd,dLSTd_aux],ignore_index= True)
        
    
    if len(dLSTn_aux)>20:
        dAEMETn = pd.concat([dAEMETn,dAEMETn_aux],ignore_index= True)
        dERA5n = pd.concat([dERA5n,dERA5n_aux],ignore_index= True)
        data_LSTn = pd.concat([data_LSTn,dLSTn_aux],ignore_index= True)

    i = i+1
    print('Estacion : {i} filtrado'.format(i = i)) 

# Calculate irradiance per hour from ssrd
dERA5d['ssrd'] = dERA5d['ssrd'].diff()
dERA5d['ssrd'] = dERA5d['ssrd']/3600

# Drop NaN or low values (<10). They are result of the diff and are nighttime values where the ssrd is not applied
dERA5d.loc[dERA5d.ssrd == np.nan, 'ssrd'] = 0
dERA5d.loc[dERA5d.ssrd <= 10, 'ssrd'] = np.nan
dERA5d = dERA5d.dropna()


# Number of hours from the begining of the period
dAEMETd["hora_rel"] = dAEMETd["Time"].apply(lambda x: (pd.to_datetime(x) - inicio).total_seconds() / 3600)
dAEMETn["hora_rel"] = dAEMETn["Time"].apply(lambda x: (pd.to_datetime(x) - inicio).total_seconds() / 3600)
dERA5d["hora_rel"] = dERA5d["valid_time"].apply(lambda x: (pd.to_datetime(x) - inicio).total_seconds() / 3600)
dERA5n["hora_rel"] = dERA5n["valid_time"].apply(lambda x: (pd.to_datetime(x) - inicio).total_seconds() / 3600)
data_LSTn["hora_rel"] = data_LSTn[["Date", "Night_view_time"]].apply(lambda x: (pd.to_datetime(x[0]) - inicio).total_seconds() / 3600 + x[1], axis=1)
data_LSTd["hora_rel"] = data_LSTd[["Date", "Day_view_time"]].apply(lambda x: (pd.to_datetime(x[0]) - inicio).total_seconds() / 3600 + x[1], axis=1)

# In situ and reanalysis datafrems for daytime and nighttime 
dAEMET_dia = pd.DataFrame()
dAEMET_nit = pd.DataFrame()

dERA5_dia = pd.DataFrame()
dERA5_nit = pd.DataFrame()

# Interpolation of in situ data for obtaining in situ values at the surpass satellite time
dAEMET_dia, dAEMET_nit = fs.interpolar_datos(dAEMETd,dAEMETn,data_LSTd,data_LSTn)

# Complementing databases with stations id, dates and hours of satellite surpass
dAEMET_dia["Fecha"] = data_LSTd['Date']
dAEMET_dia["Hora"] = data_LSTd['Day_view_time']
dAEMET_dia["INDICATIVO"] = data_LSTd['INDICATIVO']


dAEMET_nit["Fecha"] = data_LSTn['Date']
dAEMET_nit["Hora"] = data_LSTn['Night_view_time']
dAEMET_nit["INDICATIVO"] = data_LSTn['INDICATIVO']


# Interpolate reanalysis data and calculate final variables (wind, relative humidity and irradiance)
dERA5_dia,dERA5_nit = fs.datos_reanalisis()
dERA5_nit["INDICATIVO"] = data_LSTn['INDICATIVO']
dERA5_dia["INDICATIVO"] = data_LSTd['INDICATIVO']


# Creating final databases for daytime and nighttime

df_final_dia = pd.DataFrame()
df_final_nit = pd.DataFrame()

# Daytime inputs
df_final_dia['INDICATIVO'] = data_LSTd['INDICATIVO']
df_final_dia['Fecha'] = dAEMET_dia["Fecha"]
df_final_dia['Hora'] = dAEMET_dia["Hora"]

# MODIS
df_final_dia["LST_day"] = data_LSTd["LST_day"]
df_final_dia["NDVI"] = data_LSTd["NDVI"]
df_final_dia["Albedo"] = data_LSTd["Albedo"]

# ERA5
df_final_dia["Vent"] = dERA5_dia["Vent"]
df_final_dia["HR"] = dERA5_dia["HR"]
df_final_dia["ssrd"] = dERA5_dia["ssrd"]

# Geographical and topographical
df_final_dia["LATITUD"] = data_LSTd["LATITUD"]
df_final_dia["LONGITUD"] = data_LSTd["LONGITUD"]
df_final_dia["ALTITUD_DEM"] = data_LSTd["ALTITUD_DEM"]
df_final_dia["AH_DEM"] = data_LSTd["AH_DEM"]
df_final_dia["SLOPE"] = data_LSTd["SLOPE"]
df_final_dia["ASPECT"] = data_LSTd["ASPECT"]
df_final_dia["COAST"] = data_LSTd["COAST"]
df_final_dia["Zenital_angle"] = data_LSTd["Zenital_angle"]
df_final_dia["Azimutal_angle"] = data_LSTd["Azimutal_angle"]
df_final_dia["Inclinacion_solar"] = data_LSTd["Inclinacion_solar"]

# In situ data
df_final_dia['SAT_AEMET'] = dAEMET_dia["Temp"]

# Nighttime inputs
df_final_nit['INDICATIVO'] = data_LSTn['INDICATIVO']
df_final_nit['Fecha'] = dAEMET_nit["Fecha"]
df_final_nit['Hora'] = dAEMET_nit["Hora"]

# MODIS
df_final_nit["LST_night"] = data_LSTn["LST_night"]
df_final_nit["NDVI"] = data_LSTn["NDVI"]
df_final_nit["Albedo"] = data_LSTn["Albedo"]

# ERA5
df_final_nit["Vent"] = dERA5_nit["Vent"]
df_final_nit["HR"] = dERA5_nit["HR"]

# Geographical and topographical
df_final_nit["LATITUD"] = data_LSTn["LATITUD"]
df_final_nit["LONGITUD"] = data_LSTn["LONGITUD"]
df_final_nit["ALTITUD_DEM"] = data_LSTn["ALTITUD_DEM"]
df_final_nit["AH_DEM"] = data_LSTn["AH_DEM"]
df_final_nit["SLOPE"] = data_LSTn["SLOPE"]
df_final_nit["ASPECT"] = data_LSTn["ASPECT"]
df_final_nit["COAST"] = data_LSTn["COAST"]

# In situ data
df_final_nit['SAT_AEMET'] = dAEMET_nit["Temp"]

# Save final databases
os.chdir('D:\TFM\Inputs')
df_final_dia.to_csv('inputs_dia.csv', index = False)
df_final_nit.to_csv('inputs_nit.csv', index = False)