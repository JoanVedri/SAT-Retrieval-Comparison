import pandas as pd
import os
import functions_SAT_inst as fs
                                                                                                                         ##  
#Script to obtain the solar coordinates published in Valor et al. (2023) from doy, surpass hour, latitude, lonitude, slope and aspect of the pixel                                                                                                        ##


# Open MODIS data file
os.chdir('D:\TFM\APPEEARS\Dades')

df = pd.read_csv('Modis_Dia.csv',sep=',', encoding='latin-1').reset_index(drop=True)

# Change dates to datetime type
df["Date"] = pd.to_datetime(df["Date"])

# Solar coordinates calculation
df['Zenital_angle'], df['Azimutal_angle'], df['Inclinacion_solar'] = zip(*df[['Date', 'Day_view_time','LATITUD','LONGITUD','SLOPE','ASPECT']].apply(lambda x: fs.coordenades_solar(*x), axis = 1))

# Saving complete MODIS data with solar coordinates for each day
df.to_csv('Modis_Dia_CoordSolar.csv', index = False) 