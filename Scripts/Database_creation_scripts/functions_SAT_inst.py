import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
from sklearn.metrics import DistanceMetric
from cmath import pi
import math


# Functions used in this project


# coast function calculates the minimum distance between a lat, lon pair to the EEA coastline file 

def coast(lat,lon):

    # Save current directory to return to them to not interfere with main script
    current_path = os.getcwd()

    # Open DEM directory
    os.chdir('D:\TFM\DEM') 

    # Function to find the nearest coast
           
    def closest_line(point, linestrings):
        return np.argmin( [p.distance(linestring) for linestring in  coastline.geometry] )

    # Coastline vectorial file font: https://www.eea.europa.eu/data-and-maps/data/eea-coastline-for-analysis-2
    path_shape="Europe_coastline.shp" 

    # Earth radius
    RT=6371 

    # Open coastline file 
    coastline = gpd.read_file(path_shape) 
    
    # WGS84 projection
    coastline_rp=coastline.to_crs({'init': 'epsg:4326'})

    # Creating a point with stations latitude and longitude
    p = Point(lon, lat)

    # Find closest line
    closest_linestring = coastline_rp.geometry[ closest_line(p, coastline_rp.geometry) ]

    # Find closest point
    closest_point = nearest_points(p, closest_linestring)
    points = gpd.GeoDataFrame(closest_point, columns=['geometry'])

    # Convert to km
    dist = DistanceMetric.get_metric('haversine')
    points_as_floats = [ np.array([p.y, p.x]) for p in closest_point ]
    haversine_distances = dist.pairwise(np.radians(points_as_floats), np.radians(points_as_floats) ) * RT
    min_dist=haversine_distances[0][1]

    # Return to original directory
    os.chdir(current_path)

    return min_dist


# find_nearest_col_row function finds the column and row number where the latitude and longitude values in 2D-arrays latitudes and longitudes are closest to a geolocation provided in match_coords (in terms of geographical distance)

def find_nearest_col_row(latitudes, longitudes, match_coords):
  
    
    lat_dist2 = (latitudes - match_coords[0])**2
    lon_dist2 = (longitudes - match_coords[1])**2
    geo_dist2 = lat_dist2 + lon_dist2
    min_geo_dist2 = geo_dist2.min()

    # Numpy arrays are organised as (rows, cols):
    row, col = np.where(geo_dist2 == min_geo_dist2)

    # Take the first location match (there are duplicate lat/lon pairs in the WST files)
    row = int(row[0])
    col = int(col[0])

    return row, col


# find_nearest_col_row function finds the 2x2 columns and rows number where the latitude and longitude values in 2D-arrays latitudes and longitudes are closest to a geolocation provided in match_coords (in terms of geographical distance)

def find_nearest_col_row_2x2(latitudes, longitudes, match_coords):

    
    lat_dist2 = (latitudes - match_coords[0])**2
    lon_dist2 = (longitudes - match_coords[1])**2
    geo_dist2 = lat_dist2 + lon_dist2
    min_geo_dist2 = geo_dist2.min()

    # Numpy arrays are organised as (rows, cols)
    row, col = np.where(geo_dist2 == min_geo_dist2)
    # Take the first location match (there are duplicate lat/lon pairs in the WST files)
    row = int(row[0])
    col = int(col[0])
  
    # Look for closest corner
    geo_dist2=[geo_dist2[row-1,col-1], geo_dist2[row-1,col+1] , geo_dist2[row+1,col-1], geo_dist2[row+1,col+1]]
    
    pos= np.where(geo_dist2 == min(geo_dist2))[0]
    if pos ==  0: 
        x1, x2, y1, y2 = [row - 1, row, col - 1, col]
    elif pos == 1:
        x1, x2, y1, y2 = [row - 1, row, col , col + 1]
    elif pos == 2: 
        x1, x2, y1, y2 = [row, row + 1, col - 1, col]
    elif pos == 3:
        x1, x2, y1, y2 = [row, row+1, col, col + 1]

    return x1, x2, y1, y2

## coordenades_solar function calulate the azimuthal, zenithal and solar inclination angles for a given day, hour, and geograhical and topographical parameters of a pixel using Valor et al. (2023) equations

def coordenades_solar(fecha,day,fi,lon,slope,aspect):
    
    # Angular unit from º to rad 
    fi=fi*(2*math.pi)/360
    lon=lon*(2*math.pi)/360
    slope=slope*(2*math.pi)/360
    aspect=aspect*(2*math.pi)/360
    dia = fecha.dayofyear
    
    # solar variables described in Valor et al. (2023)
    x=(2*math.pi*(dia-1))/365

    d=(0.006918-0.399912*math.cos(x)+0.070257*math.sin(x)-0.006758*math.cos(2*x)+0.000907*math.sin(2*x)-0.002697*math.cos(3*x)+0.001480*math.sin(3*x))
    ET=-0.128*math.sin((360*(dia-1)/365)-2.80)-0.165*math.sin((2*360*(dia-1)/365)-19.7)
    
    LAT1=day+(lon*360/(2*math.pi*15))+ET
    omega1=15*(LAT1-12)*(2*math.pi)/360
    zenith1=math.sin(d)*math.sin(fi)+math.cos(d)*math.cos(fi)*math.cos(omega1)
    zd=math.acos(zenith1)

    azimut1=(math.cos(d)*math.sin(omega1)/math.sin(zd))
    ad=math.asin(azimut1)
    ID=math.acos(math.cos(zd)*math.cos(slope)+math.sin(zd)*math.sin(slope)*math.cos(ad-aspect))
  
    return zd,ad,ID

# interpolar_datos function interpolates temporally the in situ values of air temperature to obtain the value at the satellite surpass time

def interpolar_datos(dAEMETd,dAEMETn,data_LSTd,data_LSTn):

    s_dia = pd.Series()
    s_nit = pd.Series() 

    # Save station list
    stations_list_dia = data_LSTd['INDICATIVO'].unique()

    # Loop for interpolate the data for each station
    i = 0
    for station in stations_list_dia:
       
        s_dia_aux = pd.DataFrame()

        # In situ and LST daytima data for each station
        dAEMETd_aux = dAEMETd[dAEMETd[' INDICATIVO'] == station].reset_index(drop = True)
        data_LSTd_aux = data_LSTd[data_LSTd['INDICATIVO'] == station].reset_index(drop = True)

        # Changing time type and air temperature type to float
        horas_aemet_dia = dAEMETd_aux["hora_rel"].to_numpy(dtype="f")
        hora_lst_dia = data_LSTd_aux["hora_rel"].to_numpy(dtype="f")
        temp_dia = dAEMETd_aux["TA"].to_numpy(dtype="f")
        
        # Interpolate
        res_Temp_dia = np.interp(hora_lst_dia, horas_aemet_dia, temp_dia)

        # Change in situ temperature to Kelvin
        res_Temp_dia = res_Temp_dia + 273.15

        # Save daytime in situ data
        s_dia_aux['Temp'] = res_Temp_dia

        s_dia = pd.concat([s_dia,s_dia_aux],ignore_index= True)
        i = i+1
        print('Estacion : {i} AEMET dia'.format(i = i)) 

    
    stations_list_nit = data_LSTn['INDICATIVO'].unique()
    i = 0
    for station in stations_list_nit:
        
        s_nit_aux = pd.DataFrame()

        # In situ and LST nighttime data for each station
        dAEMETn_aux = dAEMETn[dAEMETn[' INDICATIVO'] == station].reset_index(drop = True)
        data_LSTn_aux = data_LSTn[data_LSTn['INDICATIVO'] == station].reset_index(drop = True)
        
        # Changing time type and air temperature type to float
        horas_aemet_nit = dAEMETn_aux["hora_rel"].to_numpy(dtype="f")
        hora_lst_nit = data_LSTn_aux["hora_rel"].to_numpy(dtype="f")
        temp_nit = dAEMETn_aux["TA"].to_numpy(dtype="f")

        # Interpolate
        res_Temp_nit = np.interp(hora_lst_nit, horas_aemet_nit, temp_nit)

        # Change in situ temperature to Kelvin
        res_Temp_nit = res_Temp_nit + 273.15

        # Save nighttime in situ data
        s_nit_aux['Temp'] = res_Temp_nit

        s_nit = pd.concat([s_nit,s_nit_aux],ignore_index= True)
        
        i = i+1
        print('Estacion : {i} AEMET nit'.format(i = i))    

    
    return s_dia , s_nit


# interpolar_datos_ERA5 function interpolates temporally the reanalysis values of air temperature, wind, dew point, temperature and pressure to obtain the value at the satellite surpass time

def interpolar_datos_ERA5(dERA5d,data_LSTd,dERA5n,data_LSTn):

    s_dia = pd.DataFrame()

    # Save station list
    stations_list_dia = data_LSTd['INDICATIVO'].unique()

    # Loop for interpolate data for each station
    i = 0
    for station in stations_list_dia:
       
        s_dia_aux = pd.DataFrame()

        # ERA5 and LST daytime data for each station
        dERA5d_aux = dERA5d[dERA5d['INDICATIVO'] == station].reset_index(drop = True)
        data_LSTd_aux = data_LSTd[data_LSTd['INDICATIVO'] == station].reset_index(drop = True)

        # Convert time
        s_dia_aux["time"] = data_LSTd_aux[["Date", "Day_view_time"]].apply(lambda x: pd.to_datetime(x[0]) + pd.to_timedelta(x[1], unit="h"), axis=1)
        
        # Changing variable types to float and interpolate them
        hora_lst = data_LSTd_aux["hora_rel"].to_numpy(dtype="f")
        horas_eras = dERA5d_aux["hora_rel"].to_numpy(dtype="f")
        
        # Horizontal wind
        u10 = dERA5d_aux ["u10"].to_numpy(dtype="f")
        res_u10 = np.interp(hora_lst, horas_eras, u10)

        s_dia_aux['u10'] = res_u10 

        # Vertical wind
        v10 = dERA5d_aux["v10"].to_numpy(dtype="f")
        res_v10 = np.interp(hora_lst, horas_eras, v10)

        s_dia_aux['v10'] = res_v10 

        # Dew point temperature
        d2m = dERA5d_aux["d2m"].to_numpy(dtype="f")
        res_d2m = np.interp(hora_lst, horas_eras, d2m)

        s_dia_aux['d2m'] = res_d2m

        # 2m air temperature
        t2m = dERA5d_aux["t2m"].to_numpy(dtype="f")
        res_t2m = np.interp(hora_lst, horas_eras, t2m)

        s_dia_aux['t2m'] = res_t2m

        # Pressure
        sp = dERA5d_aux["sp"].to_numpy(dtype="f")
        res_sp = np.interp(hora_lst, horas_eras, sp)

        s_dia_aux['sp'] = res_sp

        # Irradiance
        ssrd= dERA5d_aux["ssrd"].to_numpy(dtype="f")
        res_ssrd = np.interp(hora_lst, horas_eras, ssrd)

        s_dia_aux['ssrd'] = res_ssrd

        s_dia = pd.concat([s_dia,s_dia_aux],ignore_index= True)
        i = i+1
        print('Estacion : {i} ERA5 dia'.format(i = i)) 

    s_nit = pd.DataFrame()

    stations_list_nit = data_LSTn['INDICATIVO'].unique()
    i = 0
    for station in stations_list_nit:
       
        s_nit_aux = pd.DataFrame()

        # ERA5 and LST nighttime data for each station
        dERA5n_aux = dERA5n[dERA5n['INDICATIVO'] == station].reset_index(drop = True)
        data_LSTn_aux = data_LSTn[data_LSTn['INDICATIVO'] == station].reset_index(drop = True)

        s_nit_aux["time"] = data_LSTn_aux[["Date", "Night_view_time"]].apply(lambda x: pd.to_datetime(x[0]) + pd.to_timedelta(x[1], unit="h"), axis=1)
        
        # Changing variable types to float and interpolate them
        hora_lst = data_LSTn_aux["hora_rel"].to_numpy(dtype="f")
        horas_eras = dERA5n_aux["hora_rel"].to_numpy(dtype="f")

        # Horizontal wind
        u10 = dERA5n_aux["u10"].to_numpy(dtype="f")
        res_u10 = np.interp(hora_lst, horas_eras, u10)

        s_nit_aux['u10'] = res_u10 

        # Vertical wind
        v10 = dERA5n_aux["v10"].to_numpy(dtype="f")
        res_v10 = np.interp(hora_lst, horas_eras, v10)

        s_nit_aux['v10'] = res_v10 

        # Dew point temperature
        d2m = dERA5n_aux["d2m"].to_numpy(dtype="f")
        res_d2m = np.interp(hora_lst, horas_eras, d2m)

        s_nit_aux['d2m'] = res_d2m

        # 2m air temperature
        t2m = dERA5n_aux["t2m"].to_numpy(dtype="f")
        res_t2m = np.interp(hora_lst, horas_eras, t2m)

        s_nit_aux['t2m'] = res_t2m

        # Pressure
        sp = dERA5n_aux["sp"].to_numpy(dtype="f")
        res_sp = np.interp(hora_lst, horas_eras, sp)

        s_nit_aux['sp'] = res_sp

        s_nit = pd.concat([s_nit,s_nit_aux],ignore_index= True)
        i = i+1
        print('Estacion : {i} ERA5 nit'.format(i = i)) 

    return s_dia, s_nit

# datos_reanalisis function gets the ERA5 interpolated variables to calculater relative humidity (HR) and wind's modulus (Vent)

def datos_reanalisis(dERA5d, data_LSTd, dERA5n, data_LSTn):

    # interpolate ERA5 data
    df_dia, df_noche = interpolar_datos_ERA5(dERA5d, data_LSTd, dERA5n, data_LSTn)

    df_dia = df_dia.reset_index(drop=True)
    df_noche = df_noche.reset_index(drop=True)

    # Calculate nighttime relative humidity using Magnus formula
    df_noche['HR']= (np.exp(21.548-5833.0/df_noche['d2m'])/np.exp(21.548-5833.0/df_noche['t2m']))*100
    df_noche['HR'] = 100*(np.exp((17.625*(df_noche['d2m']-273.15))/(243.04+df_noche['d2m']-273.15))/ np.exp((17.625*(df_noche['t2m']-273.5))/(243.04+df_noche['t2m']-273.5)))

    # Calculate nighttime wind's modulus
    df_noche['Vent']= np.sqrt(df_noche['u10']*df_noche['u10'] + df_noche['v10']*df_noche['v10'])

    # Calculate daytime relative humidity using Magnus formula
    df_dia['HR']= (np.exp(21.548-5833.0/df_dia['d2m'])/np.exp(21.548-5833.0/df_dia['t2m']))*100
    df_dia['HR'] = 100*(np.exp((17.625*(df_dia['d2m']-273.15))/(243.04+df_dia['d2m']-273.15))/ np.exp((17.625*(df_dia['t2m']-273.5))/(243.04+df_dia['t2m']-273.5)))

    # Calculate daytime wind's modulus
    df_dia['Vent']= np.sqrt(df_dia['u10']*df_dia['u10'] + df_dia['v10']*df_dia['v10'])

    return df_dia, df_noche
