from cmath import pi
import os
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import pyproj as pp
import datetime
import io
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Script to retrive SAT from Niclos et al. (2014) equations which were obtained for the summer period over the Valencian authonomy using a stepwise method

dfd = pd.DataFrame()
dfn = pd.DataFrame()

# Definition of percentile to drop exteme high or cold temperatures 
n = 5
os.chdir('S:\TFM\Inputs')
dInputsd = pd.read_csv('inputs_dia.csv')
dInputsn = pd.read_csv('inputs_nit.csv')

# Calculate percentile
per_5 = np.percentile([dInputsd['LST_day']],n)
per_95 = np.percentile([dInputsd['LST_day']],100-n)

# Drop outliers
dInputsd = dInputsd[dInputsd['LST_day']> per_5]
dInputsd = dInputsd[dInputsd['LST_day']< per_95]

# Same with nighttime values
per_5 = np.percentile([dInputsn['LST_night']],n)
per_95 = np.percentile([dInputsn['LST_night']],100-n)

dInputsn = dInputsn[dInputsn['LST_night']> per_5]
dInputsn = dInputsn[dInputsn['LST_night']< per_95]

dInputsd = dInputsd.reset_index()
dInputsn = dInputsn.reset_index()

# Daytime inputs. Changing latitude and longitud to km. Altitudes also changed to m 
myProj = pp.Proj(proj='utm', zone=30, ellps='WGS84', units = 'km' )
G = pp.Geod(ellps='WGS84')
LST = dInputsd['LST_day']
NDVId = dInputsd['NDVI']
ALd = dInputsd['Albedo']
I = dInputsd['ssrd']/1000
HR = dInputsd['HR']
LAT = dInputsd['LATITUD']

LON = dInputsd['LONGITUD']
UTMX ,UTMY = myProj(LON,LAT)
LAT = UTMX
H = dInputsd['ALTITUD_DEM']/1000
DIST = dInputsd['COAST']
AH = dInputsd['AH_DEM']/1000
A = dInputsd['Azimutal_angle']
z = dInputsd['Zenital_angle']
s = dInputsd['SLOPE']*(np.pi/180)
i = dInputsd['Inclinacion_solar']


#Daytime equations
dfd['SATD1']=0.4*LST+178.7
dfd['SATD2']=0.26*LST-12.4*(1-NDVId)-5.9*I-0.102*HR-0.031*LAT-0.8*H-0.081*DIST+27.0*ALd+260.1
dfd['SATD3']=0.25*LST-9.3*(1-NDVId)-8.8*(1-ALd)*I-0.109*HR-0.029*LAT-2.2*H-0.057*DIST+264.6
dfd['SATD4']=0.47*LST-18.6*(1-NDVId)-6.0*I-0.04*LAT+1.9*H-0.101*DIST+44.4*ALd+195.4
dfd['SATD5']=0.49*LST-13.5*(1-NDVId)-10.5*(1-ALd)*I-0.037*LAT-0.5*H-0.056*DIST+194.6
dfd['SATD6']=LST+1.82-10.66*np.cos(z)*(1-NDVId)-0.566*A-3.72*(1-ALd)*(np.cos(i)/np.cos(z)+(np.pi - s)/np.pi)*I-3.41*AH
dfd['SATD7']=0.52*LST+152.7-8.6*np.cos(z)*(1-NDVId)+1.4*A-4.1*(1-ALd)*(np.cos(i)/np.cos(z)+(np.pi - s)/np.pi)*I -2.9*AH
dfd['SATD8']=0.52*LST+152.3-8.5*np.cos(z)*(1-NDVId)-5.4*(1-ALd)*(np.cos(i)/np.cos(z)+(np.pi - s)/np.pi)*I 
dfd['SATD9']=0.23*LST-9.5*np.cos(z)*(1-NDVId)-2.3*(1-ALd)*(np.cos(i)/np.cos(z)+(np.pi - s)/np.pi)*I-0.115*HR-0.033*LAT-2.8*H-0.059*DIST+270.4 
dfd['SATD10']=0.51*LST-15.1*np.cos(z)*(1-NDVId)-3.1*(1-ALd)*(np.cos(i)/np.cos(z)+(np.pi - s)/np.pi)*I-0.040*LAT-1*H-0.059*DIST+188.5

dfd['T. Estacio'] = dInputsd['SAT_AEMET'] 
dfdd = dfd.copy()

# Calculate the difference between retrieved SAT and in situ SAT
dfd['Dif 1'] = dfd['SATD1'] - dfd['T. Estacio']
dfd['Dif 2'] = dfd['SATD2'] - dfd['T. Estacio']
dfd['Dif 3'] = dfd['SATD3'] - dfd['T. Estacio']
dfd['Dif 4'] = dfd['SATD4'] - dfd['T. Estacio']
dfd['Dif 5'] = dfd['SATD5'] - dfd['T. Estacio']
dfd['Dif 6'] = dfd['SATD6'] - dfd['T. Estacio']
dfd['Dif 7'] = dfd['SATD7'] - dfd['T. Estacio']
dfd['Dif 8'] = dfd['SATD8'] - dfd['T. Estacio']
dfd['Dif 9'] = dfd['SATD9'] - dfd['T. Estacio']
dfd['Dif 10'] = dfd['SATD10'] - dfd['T. Estacio']

dfd.to_csv('prediccions_day_raquel.csv')


# Nighttime inputs. Changing latitude and longitud to km. Altitudes also changed to m 
LSTN = dInputsn['LST_night'].values
NDVIn = dInputsn['NDVI'].values
HRN = dInputsn['HR'].values
ALn = dInputsn['Albedo'].values
Un = dInputsn['Vent'].values

LAT = dInputsn['LATITUD']
LON = dInputsn['LONGITUD']
UTMX ,UTMY = myProj(LON,LAT)
DIST = dInputsn['COAST']
LAT = UTMX
H = dInputsn['ALTITUD_DEM']/1000

#Nighttime equations
dfn['SATN1']=0.94*LSTN+19.3
dfn['SATN2']=0.85*LSTN+1.8*(1-NDVIn)-0.129*HRN+(0.00056*HRN*HRN)-0.009*LAT-0.8*H-0.009*DIST-10.4*ALn+0.04*Un+58.9
dfn['SATN3']=0.86*LSTN+1.7*(1-NDVIn)-0.130*HRN+(0.00056*HRN*HRN)-0.009*LAT-0.7*H-0.008*DIST-10.6*ALn+56.1
dfn['SATN4']=0.92*LSTN+2.8*(1-NDVIn)-0.012*LAT+0.9*H-0.018*DIST-11.4*ALn+0.26*Un+31.8
dfn['SATN5']=0.99*LSTN+2.3*(1-NDVIn)-0.012*LAT+1.3*H-0.016*DIST-13.3*ALn+12.2

# Calculate the difference between retrieved SAT and in situ SAT
dfn['T. Estacio'] = dInputsn['SAT_AEMET']
dfnn = dfn.copy()
dfn['Dif 1'] = dfn['SATN1'] - dfn['T. Estacio']
dfn['Dif 2'] = dfn['SATN2'] - dfn['T. Estacio']
dfn['Dif 3'] = dfn['SATN3'] - dfn['T. Estacio']
dfn['Dif 4'] = dfn['SATN4'] - dfn['T. Estacio']
dfn['Dif 5'] = dfn['SATN5'] - dfn['T. Estacio']

dfn.to_csv('prediccions_night_raquel.csv')

# stats function calculates the classic statistics and robust statistics used in our paper
def stats(dif): #dif:numpy array
    minim=np.nanmin(dif)
    maxim=np.nanmax(dif)
    mn=np.nanmean(dif) #mean value (bias)
    sd=np.nanstd(dif, ddof=1) #standard deviation
    rmse=np.sqrt(sd**2+mn**2)
    mediana=np.nanmedian(dif) #median value
    rsd=np.nanmedian(np.abs(dif-mediana))*1.4826
    rrmse=np.sqrt(mediana**2+rsd**2)
    n=dif.size - np.count_nonzero(np.isnan(dif))
    return minim, maxim, mn, sd, rmse, mediana, rsd, rrmse, n

# Statistics for daytime results
statistics = pd.DataFrame()
statistics['Statistics'] = ['minim', 'maxim', 'mn', 'sd', 'rmse', 'mediana', 'rsd', 'rrmse', 'n']
statistics['Dif 1 dia'] = stats(dfd['Dif 1'].values)
statistics['Dif 2 dia'] = stats(dfd['Dif 2'].values)
statistics['Dif 3 dia'] = stats(dfd['Dif 3'].values)
statistics['Dif 4 dia'] = stats(dfd['Dif 4'].values)
statistics['Dif 5 dia'] = stats(dfd['Dif 5'].values)
statistics['Dif 6 dia'] = stats(dfd['Dif 6'].values)
statistics['Dif 7 dia'] = stats(dfd['Dif 7'].values)
statistics['Dif 8 dia'] = stats(dfd['Dif 8'].values)
statistics['Dif 9 dia'] = stats(dfd['Dif 9'].values)

# Statistics for nighttime results
statistics['Dif 1 nit'] = stats(dfn['Dif 1'].values)
statistics['Dif 2 nit'] = stats(dfn['Dif 2'].values)
statistics['Dif 3 nit'] = stats(dfn['Dif 3'].values)
statistics['Dif 4 nit'] = stats(dfn['Dif 4'].values)
statistics['Dif 5 nit'] = stats(dfn['Dif 5'].values)

# Save statistics file
statistics.to_csv('Statistics_EqRaquel.csv')


# Figure creation

# Defining ranges for axis
cbar_range=5
min_val_D =  int(np.floor(np.nanmin(dfdd.min().values) / cbar_range) * cbar_range)
max_val_D = int(np.ceil(np.nanmax(dfdd.max().values) / cbar_range) * cbar_range)

cbar_range=5
min_val_N =  int(np.floor(np.nanmin(dfnn.min().values) / cbar_range) * cbar_range)
max_val_N = int(np.ceil(np.nanmax(dfnn.max().values) / cbar_range) * cbar_range)

# Creating a line for comparing the results with 1:1 line
regression_model = LinearRegression()

plt.figure(1, figsize=(10,12))

# Contant values to fit the text
l1=[-100, 400]
cx = 20
cy = 41

cyx= 9
cyy = 2

min_val_D = 265
max_val_D = 330

plt.subplot(431)
plt.text(min_val_D + 1, max_val_D - 6, 'D.1', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_D, max_val_D, min_val_D, max_val_D])
plt.plot(dfd['T. Estacio'],dfd['SATD1'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_D, max_val_D, 10))
y= dfd['SATD1'].values
y= y.reshape(-1,1)
x= dfd['T. Estacio'].values
x= x.reshape(-1,1)
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
r2 = r2_score(y, y_predicted)
M = regression_model.coef_[0][0]
N = regression_model.intercept_[0]
x2 = np.arange(250,350)
x2= x2.reshape(-1,1)
y2 = regression_model.predict(x2)
y2= y2.reshape(-1,1)
plt.plot(x2,y2, 'k--')
plt.text(max_val_D - cx, min_val_D + cyx, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_D - cy, min_val_D + cyy, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))


plt.subplot(432)
plt.text(min_val_D + 1, max_val_D - 6, 'D.2', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_D, max_val_D, min_val_D, max_val_D])
plt.plot(dfd['T. Estacio'],dfd['SATD2'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_D, max_val_D, 10))
y= dfd['SATD2'].values
y= y.reshape(-1,1)
x= dfd['T. Estacio'].values
x= x.reshape(-1,1)
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
r2 = r2_score(y, y_predicted)
M = regression_model.coef_[0][0]
N = regression_model.intercept_[0]
x2 = np.arange(250,350)
x2= x2.reshape(-1,1)
y2 = regression_model.predict(x2)
y2= y2.reshape(-1,1)
plt.plot(x2,y2, 'k--')
plt.text(max_val_D - cx, min_val_D + cyx, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_D - cy, min_val_D + cyy, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(433)
plt.text(min_val_D + 1, max_val_D - 6, 'D.3', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_D, max_val_D, min_val_D, max_val_D])
plt.plot(dfd['T. Estacio'],dfd['SATD3'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_D, max_val_D, 10))
y= dfd['SATD3'].values
y= y.reshape(-1,1)
x= dfd['T. Estacio'].values
x= x.reshape(-1,1)
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
r2 = r2_score(y, y_predicted)
M = regression_model.coef_[0][0]
N = regression_model.intercept_[0]
x2 = np.arange(250,350)
x2= x2.reshape(-1,1)
y2 = regression_model.predict(x2)
y2= y2.reshape(-1,1)
plt.plot(x2,y2, 'k--')
plt.text(max_val_D - cx, min_val_D + cyx, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_D - cy, min_val_D + cyy, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(434)
plt.text(min_val_D + 1, max_val_D - 6, 'D.4', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_D, max_val_D, min_val_D, max_val_D])
plt.plot(dfd['T. Estacio'],dfd['SATD4'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_D, max_val_D, 10))
y= dfd['SATD4'].values
y= y.reshape(-1,1)
x= dfd['T. Estacio'].values
x= x.reshape(-1,1)
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
r2 = r2_score(y, y_predicted)
M = regression_model.coef_[0][0]
N = regression_model.intercept_[0]
x2 = np.arange(250,350)
x2= x2.reshape(-1,1)
y2 = regression_model.predict(x2)
y2= y2.reshape(-1,1)
plt.plot(x2,y2, 'k--')
plt.text(max_val_D - cx, min_val_D + cyx, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_D - cy, min_val_D + cyy, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(435)
plt.text(min_val_D + 1, max_val_D - 6, 'D.5', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_D, max_val_D, min_val_D, max_val_D])
plt.plot(dfd['T. Estacio'],dfd['SATD5'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_D, max_val_D, 10))
y= dfd['SATD5'].values
y= y.reshape(-1,1)
x= dfd['T. Estacio'].values
x= x.reshape(-1,1)
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
r2 = r2_score(y, y_predicted)
M = regression_model.coef_[0][0]
N = regression_model.intercept_[0]
x2 = np.arange(250,350)
x2= x2.reshape(-1,1)
y2 = regression_model.predict(x2)
y2= y2.reshape(-1,1)
plt.plot(x2,y2, 'k--')
plt.text(max_val_D - cx, min_val_D + cyx, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_D - cy, min_val_D + cyy, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(436)
plt.text(min_val_D + 1, max_val_D - 6, 'D.6', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_D, max_val_D, min_val_D, max_val_D])
plt.plot(dfd['T. Estacio'],dfd['SATD6'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_D, max_val_D, 10))
y= dfd['SATD6'].values
y= y.reshape(-1,1)
x= dfd['T. Estacio'].values
x= x.reshape(-1,1)
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
r2 = r2_score(y, y_predicted)
M = regression_model.coef_[0][0]
N = regression_model.intercept_[0]
x2 = np.arange(250,350)
x2= x2.reshape(-1,1)
y2 = regression_model.predict(x2)
y2= y2.reshape(-1,1)
plt.plot(x2,y2, 'k--')
plt.text(max_val_D - cx, min_val_D + cyx, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_D - cy, min_val_D + cyy, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(437)
plt.text(min_val_D + 1, max_val_D - 6, 'D.7', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_D, max_val_D, min_val_D, max_val_D])
plt.plot(dfd['T. Estacio'],dfd['SATD7'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_D, max_val_D, 10))
y= dfd['SATD7'].values
y= y.reshape(-1,1)
x= dfd['T. Estacio'].values
x= x.reshape(-1,1)
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
r2 = r2_score(y, y_predicted)
M = regression_model.coef_[0][0]
N = regression_model.intercept_[0]
x2 = np.arange(250,350)
x2= x2.reshape(-1,1)
y2 = regression_model.predict(x2)
y2= y2.reshape(-1,1)
plt.plot(x2,y2, 'k--')
plt.text(max_val_D - cx, min_val_D + cyx, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_D - cy, min_val_D + cyy, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(438)
plt.text(min_val_D + 1, max_val_D - 6, 'D.8', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_D, max_val_D, min_val_D, max_val_D])
plt.plot(dfd['T. Estacio'],dfd['SATD8'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_D, max_val_D, 10))
y= dfd['SATD8'].values
y= y.reshape(-1,1)
x= dfd['T. Estacio'].values
x= x.reshape(-1,1)
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
r2 = r2_score(y, y_predicted)
M = regression_model.coef_[0][0]
N = regression_model.intercept_[0]
x2 = np.arange(250,350)
x2= x2.reshape(-1,1)
y2 = regression_model.predict(x2)
y2= y2.reshape(-1,1)
plt.plot(x2,y2, 'k--')
plt.text(max_val_D - cx, min_val_D + cyx, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_D - cy, min_val_D + cyy, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(439)
plt.text(min_val_D + 1, max_val_D - 6, 'D.9', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_D, max_val_D, min_val_D, max_val_D])
plt.plot(dfd['T. Estacio'],dfd['SATD9'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_D, max_val_D, 10))
y= dfd['SATD9'].values
y= y.reshape(-1,1)
x= dfd['T. Estacio'].values
x= x.reshape(-1,1)
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
r2 = r2_score(y, y_predicted)
M = regression_model.coef_[0][0]
N = regression_model.intercept_[0]
x2 = np.arange(250,350)
x2= x2.reshape(-1,1)
y2 = regression_model.predict(x2)
y2= y2.reshape(-1,1)
plt.plot(x2,y2, 'k--')
plt.text(max_val_D - cx, min_val_D + cyx, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_D - cy, min_val_D + cyy, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(4,3,10)
plt.text(min_val_D + 1, max_val_D - 6, 'D.10', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_D, max_val_D, min_val_D, max_val_D])
plt.plot(dfd['T. Estacio'],dfd['SATD10'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_D, max_val_D, 10))
y= dfd['SATD10'].values
y= y.reshape(-1,1)
x= dfd['T. Estacio'].values
x= x.reshape(-1,1)
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
r2 = r2_score(y, y_predicted)
M = regression_model.coef_[0][0]
N = regression_model.intercept_[0]
x2 = np.arange(250,350)
x2= x2.reshape(-1,1)
y2 = regression_model.predict(x2)
y2= y2.reshape(-1,1)
plt.plot(x2,y2, 'k--')
plt.text(max_val_D - cx, min_val_D + cyx, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_D - cy, min_val_D + cyy, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.tight_layout()
plt.savefig('Dia_Paper.png')

cx = 17
cy = 33
min_val_N = 260
max_val_N = 315
plt.figure(2, figsize=(10,6))

plt.subplot(231)
plt.text(min_val_N + 1, max_val_N - 4, 'N.1', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_N, max_val_N, min_val_N, max_val_N])
plt.plot(dfn['T. Estacio'],dfn['SATN1'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_N, max_val_N, 10))
y= dfn['SATN1'].values
y= y.reshape(-1,1)
x= dfn['T. Estacio'].values
x= x.reshape(-1,1)
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
r2 = r2_score(y, y_predicted)
M = regression_model.coef_[0][0]
N = regression_model.intercept_[0]
x2 = np.arange(250,350)
x2= x2.reshape(-1,1)
y2 = regression_model.predict(x2)
y2= y2.reshape(-1,1)
plt.plot(x2,y2, 'k--')
plt.text(max_val_N - cx, min_val_N + 5, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_N - cy, min_val_N + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(232)
plt.text(min_val_N + 1, max_val_N - 4, 'N.2', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_N, max_val_N, min_val_N, max_val_N])
plt.plot(dfn['T. Estacio'],dfn['SATN2'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_N, max_val_N, 10))
y= dfn['SATN2'].values
y= y.reshape(-1,1)
x= dfn['T. Estacio'].values
x= x.reshape(-1,1)
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
r2 = r2_score(y, y_predicted)
M = regression_model.coef_[0][0]
N = regression_model.intercept_[0]
x2 = np.arange(250,350)
x2= x2.reshape(-1,1)
y2 = regression_model.predict(x2)
y2= y2.reshape(-1,1)
plt.plot(x2,y2, 'k--')
plt.text(max_val_N - cx, min_val_N + 5, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_N - cy, min_val_N + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(233)
plt.text(min_val_N + 1, max_val_N - 4, 'N.3', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_N, max_val_N, min_val_N, max_val_N])
plt.plot(dfn['T. Estacio'],dfn['SATN3'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_N, max_val_N, 10))
y= dfn['SATN3'].values
y= y.reshape(-1,1)
x= dfn['T. Estacio'].values
x= x.reshape(-1,1)
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
r2 = r2_score(y, y_predicted)
M = regression_model.coef_[0][0]
N = regression_model.intercept_[0]
x2 = np.arange(250,350)
x2= x2.reshape(-1,1)
y2 = regression_model.predict(x2)
y2= y2.reshape(-1,1)
plt.plot(x2,y2, 'k--')
plt.text(max_val_N - cx, min_val_N + 5, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_N - cy, min_val_N + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(234)
plt.text(min_val_N + 1, max_val_N - 4, 'N.4', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_N, max_val_N, min_val_N, max_val_N])
plt.plot(dfn['T. Estacio'],dfn['SATN4'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_N, max_val_N, 10))
y= dfn['SATN4'].values
y= y.reshape(-1,1)
x= dfn['T. Estacio'].values
x= x.reshape(-1,1)
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
r2 = r2_score(y, y_predicted)
M = regression_model.coef_[0][0]
N = regression_model.intercept_[0]
x2 = np.arange(250,350)
x2= x2.reshape(-1,1)
y2 = regression_model.predict(x2)
y2= y2.reshape(-1,1)
plt.plot(x2,y2, 'k--')
plt.text(max_val_N - cx, min_val_N + 5, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_N - cy, min_val_N + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(235)
plt.text(min_val_N + 1, max_val_N - 4, 'N.5', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_N, max_val_N, min_val_N, max_val_N])
plt.plot(dfn['T. Estacio'],dfn['SATN5'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_N, max_val_N, 10))
y= dfn['SATN5'].values
y= y.reshape(-1,1)
x= dfn['T. Estacio'].values
x= x.reshape(-1,1)
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
r2 = r2_score(y, y_predicted)
M = regression_model.coef_[0][0]
N = regression_model.intercept_[0]
x2 = np.arange(250,350)
x2= x2.reshape(-1,1)
y2 = regression_model.predict(x2)
y2= y2.reshape(-1,1)
plt.plot(x2,y2, 'k--')
plt.text(max_val_N - cx, min_val_N + 5, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_N - cy, min_val_N + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.tight_layout()
plt.savefig('Nit_Paper.png')
plt.show()