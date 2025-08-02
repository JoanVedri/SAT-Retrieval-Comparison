import numpy as np
import os
import pandas as pd
import nonlinear_methods_func as mnl
import select_nonlinear_var as vr
import scipy.stats as stats
from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split

# Script used to obtain the results of the nonlinear methods used in Vedrí et al. (2025) to retrieve the SAT

# Dataframes for predictions and statistics
prediccions_day = pd.DataFrame()
prediccions_night = pd.DataFrame()
statistics = pd.DataFrame()

# Selection of percentile to eliminate hottest and coldest SAT to avoid outliers
n = 2

# Open daytime database
os.chdir('J:\Joan')
df_dia = pd.read_csv('inputs_dia.csv',sep=',', encoding='latin-1').reset_index(drop=True)
os.chdir('J:\Joan\ANolineal')

# Calculate percentiles
per_5 = np.percentile([df_dia['LST_day']],n)
per_95 = np.percentile([df_dia['LST_day']],100-n)

# Drop outliers
df_dia = df_dia[df_dia['LST_day']> per_5]
df_dia = df_dia[df_dia['LST_day']< per_95]

# Changing units to use small numbers (good practice for some of the methods used in this script)
df_dia['ssrd'] = df_dia['ssrd']/1000
df_dia['ALTITUD_DEM'] = df_dia['ALTITUD_DEM']/1000
df_dia['AH_DEM'] = df_dia['AH_DEM']/1000


# Creating input variables dataframe and target dataframe 
X = df_dia.drop(['Fecha','Hora','SAT_AEMET'],axis='columns')
Y = pd.DataFrame()

Y['SAT_AEMET'] = df_dia['SAT_AEMET']


# Train test split with 70/30 holdout
Xtrain_day, Xtest_day, ytrain_day, ytest_day = train_test_split(X, Y, test_size=0.3, random_state= 1234,shuffle= True)

# Selecting variables to train
Xtrain_day, Xtest_day = vr.dtotes(Xtrain_day,Xtest_day)

prediccions_day['id'] = Xtest_day['INDICATIVO']

# Saving columns name for features selection
Xtrain_day = Xtrain_day.drop(['INDICATIVO'],axis='columns')
Xtest_day = Xtest_day.drop(['INDICATIVO'],axis='columns')
col = Xtrain_day.columns

# Save test data
prediccions_day['SAT'] = ytest_day

# Train models and extracting the predictions
prediccio = mnl.RF_day(Xtrain_day,ytrain_day,Xtest_day,ytest_day, col,time ='day')

prediccions_day['RF'] = prediccio

prediccio = mnl.XGB_day(Xtrain_day,ytrain_day,Xtest_day,ytest_day,col,time ='day')

prediccions_day['XGB'] = prediccio

prediccio = mnl.KNN(Xtrain_day,ytrain_day,Xtest_day,ytest_day,time ='day')

prediccions_day['KNN'] = prediccio

prediccio = mnl.MLP_day(Xtrain_day,ytrain_day,Xtest_day,ytest_day)

prediccions_day['MLP'] = prediccio


# Repeating the process for nighttime models
os.chdir('J:\Joan')
df_nit = pd.read_csv('inputs_nit.csv',sep=',', encoding='latin-1').reset_index(drop=True)
os.chdir('J:\Joan\ANolineal')

per_5 = np.percentile([df_nit['LST_night']],n)
per_95 = np.percentile([df_nit['LST_night']],100-n)


df_nit = df_nit[df_nit['LST_night']> per_5]
df_nit = df_nit[df_nit['LST_night']< per_95]


X = df_nit.drop(['Fecha','Hora','SAT_AEMET'],axis='columns')

Y = df_nit['SAT_AEMET']

df_nit['ALTITUD_DEM'] = df_nit['ALTITUD_DEM']/1000
df_nit['AH_DEM'] = df_nit['AH_DEM']/1000

Xtrain_night, Xtest_night, ytrain_night, ytest_night = train_test_split(X, Y, test_size=0.3, random_state= 1234,shuffle= True) 

Xtrain_night, Xtest_night = vr.ntotes(Xtrain_night, Xtest_night)

prediccions_night['id'] = Xtest_night['INDICATIVO']

Xtrain_night = Xtrain_night.drop(['INDICATIVO'],axis='columns')
Xtest_night = Xtest_night.drop(['INDICATIVO'],axis='columns')

col = Xtest_night.columns


prediccions_night['SAT'] = ytest_night

prediccio = mnl.RF_night(Xtrain_night,ytrain_night,Xtest_night,ytest_night,col,time ='night')
prediccions_night['RF'] = prediccio

prediccio = mnl.XGB_night(Xtrain_night,ytrain_night,Xtest_night,ytest_night,col,time ='night')
prediccions_night['XGB'] = prediccio

prediccio = mnl.KNN(Xtrain_night,ytrain_night,Xtest_night,ytest_night,time ='night')
prediccions_night['KNN'] = prediccio

prediccio = mnl.MLP_night(Xtrain_night,ytrain_night,Xtest_night,ytest_night)
prediccions_night['MLP'] = prediccio

# Calculate diff between predictions and real values
prediccions_day['Dif_RF'] = prediccions_day['RF'] - prediccions_day['SAT']
prediccions_day['Dif_XGB'] = prediccions_day['XGB'] - prediccions_day['SAT']
prediccions_day['Dif_KNN'] = prediccions_day['KNN'] - prediccions_day['SAT']
prediccions_day['Dif_MLP'] = prediccions_day['MLP'] - prediccions_day['SAT']


prediccions_night['Dif_RF'] = prediccions_night['RF'] - prediccions_night['SAT']
prediccions_night['Dif_XGB'] = prediccions_night['XGB'] - prediccions_night['SAT']
prediccions_night['Dif_KNN'] = prediccions_night['KNN'] - prediccions_night['SAT']
prediccions_night['Dif_MLP'] = prediccions_night['MLP'] - prediccions_night['SAT']

# Save differences
prediccions_day.to_csv('prediccions_day_nolineals.csv')
prediccions_night.to_csv('prediccions_night_nolineals.csv')


# Creating figure same than linear equations script

cbar_range=5
min_val_D =  int(np.floor(260 / cbar_range) * cbar_range)
max_val_D = int(np.ceil(320 / cbar_range) * cbar_range)

cbar_range=5
min_val_N =  int(np.floor(260/ cbar_range) * cbar_range)
max_val_N = int(np.ceil(305 / cbar_range) * cbar_range)

regression_model = LinearRegression()

cx = 14
cy = 25

plt.figure(3, figsize=(10,6))

l1=[-100, 400]

plt.subplot(221)
x = prediccions_day['SAT'].values
y = prediccions_day['RF'].values 
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

plt.text(min_val_D + 1, max_val_D - 5, 'a) RF', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_D, max_val_D, min_val_D, max_val_D])
plt.plot(prediccions_day['SAT'],prediccions_day['RF'], c = z,marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_D, max_val_D, 10))
y= prediccions_day['RF'].values 
y= y.reshape(-1,1)
x= prediccions_day['SAT'].values
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
plt.text(max_val_D - cx, min_val_D + 5, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_D - cy, min_val_D + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(222)
plt.text(min_val_D + 1, max_val_D - 5, 'b) XGB', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_D, max_val_D, min_val_D, max_val_D])
plt.plot(prediccions_day['SAT'],prediccions_day['XGB'], '#ff6961',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_D, max_val_D, 10))
y= prediccions_day['XGB'].values 
y= y.reshape(-1,1)
x= prediccions_day['SAT'].values
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
plt.text(max_val_D - cx, min_val_D + 5, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_D - cy, min_val_D + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(223)
plt.text(min_val_D + 1, max_val_D - 5, 'c) KNN', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_D, max_val_D, min_val_D, max_val_D])
plt.plot(prediccions_day['SAT'],prediccions_day['KNN'], '#ff6961',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_D, max_val_D, 10))
y= prediccions_day['KNN'].values 
y= y.reshape(-1,1)
x= prediccions_day['SAT'].values
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
plt.text(max_val_D - cx, min_val_D + 5, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_D - cy, min_val_D + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(224)
plt.text(min_val_D + 1, max_val_D - 5, 'd) MLP', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_D, max_val_D, min_val_D, max_val_D])
plt.plot(prediccions_day['SAT'],prediccions_day['MLP'], '#ff6961',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_D, max_val_D, 10))
y= prediccions_day['MLP'].values 
y= y.reshape(-1,1)
x= prediccions_day['SAT'].values
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
plt.text(max_val_D - cx, min_val_D + 5, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_D - cy, min_val_D + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.tight_layout()
plt.savefig('Dia.png')

cx = 10
cy = 18
plt.figure(4, figsize=(10,6))

plt.subplot(221)
plt.text(min_val_N + 1, max_val_N - 3, 'a) RF', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_N, max_val_N, min_val_N, max_val_N])
plt.plot(prediccions_night['SAT'],prediccions_night['RF'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_N, max_val_N, 10))
y= prediccions_night['RF'].values
y= y.reshape(-1,1)
x= prediccions_night['SAT'].values
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
plt.text(max_val_N - cx, min_val_N + 4, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_N - cy, min_val_N + 4, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))
plt.text(max_val_N - cy, min_val_N + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(222)
plt.text(min_val_N + 1, max_val_N - 3, 'b) XGB', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_N, max_val_N, min_val_N, max_val_N])
plt.plot(prediccions_night['SAT'],prediccions_night['XGB'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_N, max_val_N, 10))
y= prediccions_night['XGB'].values
y= y.reshape(-1,1)
x= prediccions_night['SAT'].values
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
plt.text(max_val_N - cx, min_val_N + 4, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_N - cy, min_val_N + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(223)
plt.text(min_val_N + 1, max_val_N - 3, 'c) KNN', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_N, max_val_N, min_val_N, max_val_N])
plt.plot(prediccions_night['SAT'],prediccions_night['KNN'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_N, max_val_N, 10))
y= prediccions_night['KNN'].values
y= y.reshape(-1,1)
x= prediccions_night['SAT'].values
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
plt.text(max_val_N - cx, min_val_N + 4, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_N - cy, min_val_N + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(224)
plt.text(min_val_N + 1, max_val_N - 3, 'd) MLP', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_N, max_val_N, min_val_N, max_val_N])
plt.plot(prediccions_night['SAT'],prediccions_night['MLP'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_N, max_val_N, 10))
y= prediccions_night['MLP'].values
y= y.reshape(-1,1)
x= prediccions_night['SAT'].values
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
plt.text(max_val_N - cx, min_val_N + 4, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_N - cy, min_val_N + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.tight_layout()
plt.savefig('Nit.png')


dif_day = pd.DataFrame()
dif_night = pd.DataFrame()

dif_day['Dif RF'] = prediccions_day['RF'] - prediccions_day['SAT']
dif_day['Dif XGB'] = prediccions_day['XGB'] - prediccions_day['SAT']
dif_day['Dif KNN'] = prediccions_day['KNN'] - prediccions_day['SAT']
dif_day['Dif MLP'] = prediccions_day['MLP'] - prediccions_day['SAT']


dif_night['Dif RF'] = prediccions_night['RF'] - prediccions_night['SAT']
dif_night['Dif XGB'] = prediccions_night['XGB'] - prediccions_night['SAT']
dif_night['Dif KNN'] = prediccions_night['KNN'] - prediccions_night['SAT']
dif_night['Dif MLP'] = prediccions_night['MLP'] - prediccions_night['SAT']


# Function to calculate classical statistics and robust statistics
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

# Calculate statistics and save them
statistics = pd.DataFrame()
statistics['Statistics'] = ['minim', 'maxim', 'mn', 'sd', 'rmse', 'mediana', 'rsd', 'rrmse', 'n']

statistics['Dif RF Day'] = stats(dif_day['Dif RF'].values)
statistics['Dif XGB Day'] = stats(dif_day['Dif XGB'].values)
statistics['Dif KNN Day'] = stats(dif_day['Dif KNN'].values)
statistics['Dif MLP Day'] = stats(dif_day['Dif MLP'].values)


statistics['Dif RF Night'] = stats(dif_night['Dif RF'].values)
statistics['Dif XGB Night'] = stats(dif_night['Dif XGB'].values)
statistics['Dif KNN Night'] = stats(dif_night['Dif KNN'].values)
statistics['Dif MLP Night'] = stats(dif_night['Dif MLP'].values)


statistics.to_csv('statistics_nolineals.csv', index=False)