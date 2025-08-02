import numpy as np
import os
import pandas as pd
import linear_methods_func as m
import select_linear_var as vr
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Script used to obtain the results of the linear methods used in Vedrí et al. (2025) to retrieve the SAT

# Dataframes creation for predictions and statistics
prediccions_day = pd.DataFrame()
prediccions_night = pd.DataFrame()
statistics = pd.DataFrame()

# Selection of percentile to eliminate hottest and coldest SAT to avoid outliers
n = 2

# Open daytime database

os.chdir('J:\Joan')
df_dia = pd.read_csv('inputs_dia.csv',sep=',', encoding='latin-1').reset_index(drop=True)
os.chdir('J:\Joan\lineal')

# Calculate percentiles
per_5 = np.percentile([df_dia['LST_day']],n)
per_95 = np.percentile([df_dia['LST_day']],100-n)

# Drop outliers
df_dia = df_dia[df_dia['LST_day']> per_5]
df_dia = df_dia[df_dia['LST_day']< per_95]

# Changing units to homogenize coeficients with previous studies
df_dia['ssrd'] = df_dia['ssrd']/1000
df_dia['ALTITUD_DEM'] = df_dia['ALTITUD_DEM']/1000
df_dia['AH_DEM'] = df_dia['AH_DEM']/1000


# Creating input variables dataframe and target dataframe 
X = df_dia.drop(['INDICATIVO','Fecha','Hora','SAT_AEMET'],axis='columns')
Y = pd.DataFrame()

Y['SAT_AEMET'] = df_dia['SAT_AEMET']


# Train test split with 70/30 holdout
Xtrain_day, Xtest_day, ytrain_day, ytest_day = train_test_split(X, Y, test_size=0.3, random_state= 1234,shuffle= True)

# Selecting variables to train
Xtrain_day, Xtest_day = vr.dtotes(Xtrain_day,Xtest_day)

# Save test data
prediccions_day['SAT'] = ytest_day

# Train models and extracting the predictions, only OLS coeficients are saved because of results (Vedrí et al. (2025))
prediccio, coeficient_d_mlr = m.mlr(Xtrain_day,ytrain_day,Xtest_day,ytest_day)

prediccions_day['MLR'] = prediccio

coeficient_d_mlr.to_csv('coef_day_mlr.csv')

prediccio = m.ridge(Xtrain_day,ytrain_day,Xtest_day,ytest_day,time = 'day')

prediccions_day['Ridge'] = prediccio

prediccio  = m.lasso(Xtrain_day,ytrain_day,Xtest_day,ytest_day,time = 'day')

prediccions_day['Lasso'] = prediccio


prediccio = m.EN(Xtrain_day,ytrain_day,Xtest_day,ytest_day,time = 'day')

prediccions_day['EN'] = prediccio


# Repeating the process for nighttime models
os.chdir('J:\Joan')
df_nit = pd.read_csv('inputs_nit.csv',sep=',', encoding='latin-1').reset_index(drop=True)
os.chdir('J:\Joan\lineal')

per_5 = np.percentile([df_nit['LST_night']],n)
per_95 = np.percentile([df_nit['LST_night']],100-n)


df_nit = df_nit[df_nit['LST_night']> per_5]
df_nit = df_nit[df_nit['LST_night']< per_95]


df_nit['ALTITUD_DEM'] = df_nit['ALTITUD_DEM']/1000
df_nit['AH_DEM'] = df_nit['AH_DEM']/1000


X = df_nit.drop(['INDICATIVO','Fecha','Hora','SAT_AEMET'],axis='columns')

Y = df_nit['SAT_AEMET'] 



Xtrain_night, Xtest_night, ytrain_night, ytest_night = train_test_split(X, Y, test_size=0.3, random_state= 1234,shuffle= True) 

Xtrain_night, Xtest_night = vr.ntotes(Xtrain_night, Xtest_night)

prediccions_night['SAT'] = ytest_night

prediccio, coeficient_n_mlr = m.mlr(Xtrain_night,ytrain_night,Xtest_night,ytest_night)

prediccions_night['MLR'] = prediccio

coeficient_n_mlr.to_csv('coef_nit_mlr.csv')

prediccio = m.ridge(Xtrain_night,ytrain_night,Xtest_night,ytest_night,time = 'night')

prediccions_night['Ridge'] = prediccio

prediccio = m.lasso(Xtrain_night,ytrain_night,Xtest_night,ytest_night,time = 'night')

prediccions_night['Lasso'] = prediccio

prediccio= m.EN(Xtrain_night,ytrain_night,Xtest_night,ytest_night, time = 'night')

prediccions_night['EN'] = prediccio

# Save predictions

prediccions_day.to_csv('pred_dia.csv',index=False)
prediccions_night.to_csv('pred_nit.csv',index=False)

# Calculate diff between predictions and real values
prediccions_day['Dif_MLR'] = prediccions_day['MLR'] - prediccions_day['SAT']
prediccions_day['Dif_Ridge'] = prediccions_day['Ridge'] - prediccions_day['SAT']
prediccions_day['Dif_Lasso'] = prediccions_day['Lasso'] - prediccions_day['SAT']
prediccions_day['Dif_EN'] = prediccions_day['EN'] - prediccions_day['SAT']

prediccions_night['Dif_MLR'] = prediccions_night['MLR'] - prediccions_night['SAT']
prediccions_night['Dif_Ridge'] = prediccions_night['Ridge'] - prediccions_night['SAT']
prediccions_night['Dif_Lasso'] = prediccions_night['Lasso'] - prediccions_night['SAT']
prediccions_night['Dif_EN'] = prediccions_night['EN'] - prediccions_night['SAT']

# Save differences
os.chdir("J:\Joan\prediccions")
prediccions_day.to_csv('prediccions_day_lineal.csv')
prediccions_night.to_csv('prediccions_night_lineal.csv')

os.chdir('J:\Joan\lineal')

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

plt.figure(1, figsize=(10,6))

l1=[-100, 400]


plt.subplot(221)
plt.text(min_val_D + 1, max_val_D - 5, 'a) OLS', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_D, max_val_D, min_val_D, max_val_D])
plt.plot(prediccions_day['SAT'],prediccions_day['MLR'], '#ff6961',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_D, max_val_D, 10))
y= prediccions_day['MLR'].values 
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
plt.grid(None)
plt.plot(x2,y2, 'k--')
plt.text(max_val_D - cx, min_val_D + 5, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_D - cy, min_val_D + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(222)
plt.text(min_val_D + 1, max_val_D - 5, 'b) RIDGE', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_D, max_val_D, min_val_D, max_val_D])
plt.plot(prediccions_day['SAT'],prediccions_day['Ridge'], '#ff6961',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_D, max_val_D, 10))
y= prediccions_day['Ridge'].values 
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
plt.grid(None)
plt.text(max_val_D - cx, min_val_D + 5, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_D - cy, min_val_D + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))


plt.subplot(223)
plt.text(min_val_D + 1, max_val_D - 5, 'c) LASSO', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_D, max_val_D, min_val_D, max_val_D])
plt.plot(prediccions_day['SAT'],prediccions_day['Lasso'], '#ff6961',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_D, max_val_D, 10))
y= prediccions_day['Lasso'].values 
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
plt.grid(None)
plt.text(max_val_D - cx, min_val_D + 5, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_D - cy, min_val_D + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(224)
plt.text(min_val_D + 1, max_val_D - 5, 'd) EN', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_D, max_val_D, min_val_D, max_val_D])
plt.plot(prediccions_day['SAT'],prediccions_day['EN'], '#ff6961',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_D, max_val_D, 10))
y= prediccions_day['EN'].values 
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
plt.grid(None)
plt.text(max_val_D - cx, min_val_D + 5, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_D - cy, min_val_D + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))


plt.tight_layout()
plt.show()

cx = 10
cy = 18
plt.figure(2, figsize=(10,6))

plt.subplot(221)
plt.text(min_val_N + 1, max_val_N - 4, 'a) OLS', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_N, max_val_N, min_val_N, max_val_N])
plt.plot(prediccions_night['SAT'],prediccions_night['MLR'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_N, max_val_N, 10))
y= prediccions_night['MLR'].values
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
plt.grid(None)
plt.text(max_val_N - cx, min_val_N + 4, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_N - cy, min_val_N + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(222)
plt.text(min_val_N + 1, max_val_N - 4, 'b) RIDGE', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_N, max_val_N, min_val_N, max_val_N])
plt.plot(prediccions_night['SAT'],prediccions_night['Ridge'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_N, max_val_N, 10))
y= prediccions_night['Ridge'].values
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
plt.grid(None)
plt.text(max_val_N - cx, min_val_N + 4, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_N - cy, min_val_N + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(223)
plt.text(min_val_N + 1, max_val_N - 4, 'c) LASSO', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_N, max_val_N, min_val_N, max_val_N])
plt.plot(prediccions_night['SAT'],prediccions_night['Lasso'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_N, max_val_N, 10))
y= prediccions_night['Lasso'].values
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
plt.grid(None)
plt.text(max_val_N - cx, min_val_N + 4, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_N - cy, min_val_N + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.subplot(224)
plt.text(min_val_N + 1, max_val_N - 4, 'd) EN', weight = 'bold')
plt.ylabel('T$_{Model}$ (K)')
plt.xlabel('T$_{Station}$ (K)')
plt.axis([min_val_N, max_val_N, min_val_N, max_val_N])
plt.plot(prediccions_night['SAT'],prediccions_night['EN'], '#2066a8',marker = 'o',linestyle='None')
plt.plot(l1,l1,'k-')
plt.yticks(range(min_val_N, max_val_N, 10))
y= prediccions_night['EN'].values
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
plt.grid(None)
plt.text(max_val_N - cx, min_val_N + 4, 'R$^2$ = {:2.2f}'.format(r2) )
plt.text(max_val_N - cy, min_val_N + 1, 'T$_S$ = {:.2f}T$_E$ + {:.2f}'.format(M,N))

plt.tight_layout()
plt.show()

dif_day = pd.DataFrame()
dif_night = pd.DataFrame()

dif_day['Dif MLR'] = prediccions_day['MLR'] - prediccions_day['SAT']
dif_day['Dif Ridge'] = prediccions_day['Ridge'] - prediccions_day['SAT']
dif_day['Dif Lasso'] = prediccions_day['Lasso'] - prediccions_day['SAT']
dif_day['Dif EN'] = prediccions_day['EN'] - prediccions_day['SAT']


dif_night['Dif MLR'] = prediccions_night['MLR'] - prediccions_night['SAT']
dif_night['Dif Ridge'] = prediccions_night['Ridge'] - prediccions_night['SAT']
dif_night['Dif Lasso'] = prediccions_night['Lasso'] - prediccions_night['SAT']
dif_night['Dif EN'] = prediccions_night['EN'] - prediccions_night['SAT']

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
statistics['Dif MLR Day'] = stats(dif_day['Dif MLR'].values)
statistics['Dif Ridge Day'] = stats(dif_day['Dif Ridge'].values)
statistics['Dif Lasso Day'] = stats(dif_day['Dif Lasso'].values)
statistics['Dif EN Day'] = stats(dif_day['Dif EN'].values)

statistics['Dif MLR Night'] = stats(dif_night['Dif MLR'].values)
statistics['Dif Ridge Night'] = stats(dif_night['Dif Ridge'].values)
statistics['Dif Lasso Night'] = stats(dif_night['Dif Lasso'].values)
statistics['Dif EN Night'] = stats(dif_night['Dif EN'].values)


statistics.to_csv('statistics_lineals.csv', index=False)
