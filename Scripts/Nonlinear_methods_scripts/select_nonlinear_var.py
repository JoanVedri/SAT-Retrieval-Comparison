import pandas as pd

# Script to select which variables do you want to use from the whole database to train the models

# dtotes works for daytime variables
def dtotes(Xtrain_day,Xtest_day):

    Xtrain = pd.DataFrame()
    Xtest = pd. DataFrame()

    Xtrain['INDICATIVO'] = Xtrain_day['INDICATIVO']
    Xtrain['LST_day'] = Xtrain_day['LST_day']
    Xtrain['NDVI'] = Xtrain_day['NDVI']
    Xtrain['Albedo'] = Xtrain_day['Albedo']
    Xtrain['Vent'] = Xtrain_day['Vent']
    Xtrain['HR'] = Xtrain_day['HR']
    Xtrain['ssrd'] = Xtrain_day['ssrd']
    Xtrain['LATITUD'] = Xtrain_day['LATITUD']
    Xtrain['LONGITUD'] = Xtrain_day['LONGITUD']
    Xtrain['ALTITUD_DEM'] = Xtrain_day['ALTITUD_DEM']
    Xtrain['AH_DEM'] = Xtrain_day['AH_DEM']
    Xtrain['SLOPE'] = Xtrain_day['SLOPE']
    Xtrain['ASPECT'] = Xtrain_day['ASPECT']
    Xtrain['COAST'] = Xtrain_day['COAST']
    Xtrain['Zenital_angle'] = Xtrain_day['Zenital_angle']
    Xtrain['Azimutal_angle'] = Xtrain_day['Azimutal_angle'] 
    Xtrain['Inclinacion_solar'] = Xtrain_day['Inclinacion_solar']
    
    Xtest['INDICATIVO'] = Xtest_day['INDICATIVO']
    Xtest['LST_day'] = Xtest_day['LST_day']
    Xtest['NDVI'] = Xtest_day['NDVI']
    Xtest['Albedo'] = Xtest_day['Albedo']
    Xtest['Vent'] = Xtest_day['Vent']
    Xtest['HR'] = Xtest_day['HR']
    Xtest['ssrd'] = Xtest_day['ssrd']
    Xtest['LATITUD'] = Xtest_day['LATITUD']
    Xtest['LONGITUD'] = Xtest_day['LONGITUD']
    Xtest['ALTITUD_DEM'] = Xtest_day['ALTITUD_DEM']
    Xtest['AH_DEM'] = Xtest_day['AH_DEM']
    Xtest['SLOPE'] = Xtest_day['SLOPE']
    Xtest['ASPECT'] = Xtest_day['ASPECT']
    Xtest['COAST'] = Xtest_day['COAST']
    Xtest['Zenital_angle'] = Xtest_day['Zenital_angle']
    Xtest['Azimutal_angle'] = Xtest_day['Azimutal_angle'] 
    Xtest['Inclinacion_solar'] = Xtest_day['Inclinacion_solar']


    return Xtrain,Xtest

# ntotes works for nighttime variables
def ntotes(Xtrain_night,Xtest_night):
# Introducció de dates dia
    Xtrain = pd.DataFrame()
    Xtest = pd. DataFrame()
    
    Xtrain['INDICATIVO'] = Xtrain_night['INDICATIVO']
    Xtrain['LST_night'] = Xtrain_night['LST_night']
    Xtrain['NDVI'] = Xtrain_night['NDVI']
    Xtrain['Albedo'] =Xtrain_night['Albedo']
    Xtrain['Vent'] = Xtrain_night['Vent']
    Xtrain['HR'] = Xtrain_night['HR']
    Xtrain['LATITUD'] = Xtrain_night['LATITUD']
    Xtrain['LONGITUD'] = Xtrain_night['LONGITUD']
    Xtrain['ALTITUD_DEM'] = Xtrain_night['ALTITUD_DEM']
    Xtrain['AH_DEM'] = Xtrain_night['AH_DEM']
    Xtrain['SLOPE'] = Xtrain_night['SLOPE']
    Xtrain['ASPECT'] = Xtrain_night['ASPECT']
    Xtrain['COAST'] = Xtrain_night['COAST']

    Xtest['INDICATIVO'] = Xtest_night['INDICATIVO']
    Xtest['LST_night'] = Xtest_night['LST_night']
    Xtest['NDVI'] = Xtest_night['NDVI']
    Xtest['Albedo'] =Xtest_night['Albedo']
    Xtest['Vent'] = Xtest_night['Vent']
    Xtest['HR'] = Xtest_night['HR']
    Xtest['LATITUD'] = Xtest_night['LATITUD']
    Xtest['LONGITUD'] = Xtest_night['LONGITUD']
    Xtest['ALTITUD_DEM'] = Xtest_night['ALTITUD_DEM']
    Xtest['AH_DEM'] = Xtest_night['AH_DEM']
    Xtest['SLOPE'] = Xtest_night['SLOPE']
    Xtest['ASPECT'] = Xtest_night['ASPECT']
    Xtest['COAST'] = Xtest_night['COAST']


    return Xtrain,Xtest


