import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from keras import Sequential
import pandas as pd
import plotly.express as px
import keras

# Script with the nonlinear methods used in Vedrí et al. (2025), some methods are duplicated due to improve the optimization of the parameters

# Random forest method for daytime
def RF_day(Xtrain,ytrain,Xtest, ytest,col,time):

   # Define the hyperparameter distributions
    param_dist = {
        'max_depth': stats.randint(30, 40),
        'n_estimators':stats.randint(450, 550),
    }

    # Create a based model
    rfd = RandomForestRegressor()

    # Instantiate the grid search model
    random_searchd = RandomizedSearchCV(rfd, param_distributions=param_dist, n_iter=10, cv=10, scoring='neg_mean_squared_error',verbose=1,random_state=1234, n_jobs=60)
    
    # Fit the RandomizedSearchCV object to the training data
    random_searchd.fit(Xtrain, ytrain.values.ravel())

    # Print the best set of hyperparameters and the corresponding score
    f = open ('parametersRF_{time}.txt'.format(time = time),'w')
    f.write("Best set of hyperparameters RF {time}: {res}".format(time = time, res = random_searchd.best_params_))
    f.close()

    # Use best parameters
    regressor = random_searchd.best_estimator_

    # Making predictions on the same data or new data
    prediccion_RF = regressor.predict(Xtest)


    # Feature importance from the XGBoost model
    importances = regressor.feature_importances_
    features_sel= col
    indices = np.argsort(importances)

    plt.figure(1, figsize=(10,6))
    plt.title('Feature Importances Random Forest')
    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [features_sel[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.xlim(0, 0.9)
    plt.savefig('importance_rf_{time}.png'.format(time = time))
    plt.close()

    results = pd.DataFrame(random_searchd.cv_results_)


    fig = px.scatter(
        results,
        x='param_n_estimators',
        y= 'param_max_depth',
        color='mean_test_score',
        labels={'param_n_estimators': 'Number of Estimators', 'param_max_depth': 'Max Depth'},
        title='Hyperparameter Tuning Results'
    )

    # Customize the plot layout
    fig.update_layout(
        coloraxis_colorbar=dict(title='Mean Test Score'),
        height=600,
        width=800,
        showlegend=False
    )

    # Show the plot
    fig.show()
    return prediccion_RF


# Random forest method for nighttime
def RF_night(Xtrain,ytrain,Xtest, ytest,col,time):

    # Define the hyperparameter distributions
    param_dist = {
        'max_depth': stats.randint(28, 35),
        'n_estimators':stats.randint(450, 550),
    }   
    # Create a based model
    rfn = RandomForestRegressor()
    # Instantiate the grid search model
    random_searchn = RandomizedSearchCV(rfn, param_distributions=param_dist, n_iter=10, cv=10, scoring='neg_mean_squared_error',verbose=1,random_state=1234,n_jobs=60)
    
    # Fit the regressor with x and y data
    # Fit the RandomizedSearchCV object to the training data
    random_searchn.fit(Xtrain, ytrain.values.ravel())

    # Print the best set of hyperparameters and the corresponding score
    f = open ('parametersRF_{time}.txt'.format(time = time),'w')
    f.write("Best set of hyperparameters RF {time}: {res}".format(time = time, res = random_searchn.best_params_))
    f.close()

    # Use best parameters
    regressor = random_searchn.best_estimator_

    # Making predictions on the same data or new data
    prediccion_RF = regressor.predict(Xtest)


    # Feature importance from the XGBoost model
    importances = regressor.feature_importances_
    features_sel= col
    indices = np.argsort(importances)

    plt.figure(1, figsize=(10,6))
    plt.title('Feature Importances Random Forest')
    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [features_sel[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.xlim(0, 0.9)
    plt.savefig('importance_rf_{time}.png'.format(time = time))
    plt.close()

    results = pd.DataFrame(random_searchn.cv_results_)

    fig = px.scatter(
        results,
        x='param_n_estimators',
        y= 'param_max_depth',
        color='mean_test_score',
        labels={'param_n_estimators': 'Number of Estimators', 'param_max_depth': 'Max Depth'},
        title='Hyperparameter Tuning Results'
    )

    # Customize the plot layout
    fig.update_layout(
        coloraxis_colorbar=dict(title='Mean Test Score'),
        height=600,
        width=800,
        showlegend=False
    )

    # Show the plot
    fig.show()


    return prediccion_RF



# XGBoost method for daytime
def XGB_day(Xtrain,ytrain,Xtest, ytest,col,time):
    

    # Define the hyperparameter distributions
    param_dist = {
        'max_depth': stats.randint(5, 7),
        'learning_rate': stats.uniform(0.02, 0.06),
        'n_estimators':stats.randint(1400, 1700),
        'subsample': stats.uniform(0.5, 0.5),
    }


    # Create the XGBoost model object
    xgb_modeld = xgb.XGBRegressor()

    # Create the RandomizedSearchCV object
    random_searchd = RandomizedSearchCV(xgb_modeld, param_distributions=param_dist, n_iter=10, cv=10, scoring='neg_mean_squared_error',verbose=1,random_state=1234,n_jobs=60)

    # Fit the RandomizedSearchCV object to the training data
    random_searchd.fit(Xtrain, ytrain)

    f = open ('parametersXGB_{time}.txt'.format(time = time),'w')
    f.write("Best set of hyperparameters XGB {time}: {res}".format(time = time, res = random_searchd.best_params_))
    f.close()

    # Use best parameters
    model = random_searchd.best_estimator_

    model.save_model("XGB_SAT_DAY_noah.json")

    # Making predictions on the same data or new data
    prediccion_XGB = model.predict(Xtest)

    # Feature importance from the XGBoost model
    importances = model.feature_importances_

    features_sel= col
    indices = np.argsort(importances)

    plt.figure(1, figsize=(10,6))
    plt.title('Feature Importances XGBoost')
    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [features_sel[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.xlim(0, 0.9)
    plt.tight_layout()
    plt.savefig('importance_xgboost_{time}.png'.format(time = time))
    plt.close()

    results = pd.DataFrame(random_searchd.cv_results_)

    fig = px.scatter_3d(
        results,
        x='param_n_estimators',
        y= 'param_max_depth',
        z = 'param_learning_rate',
        # size='mean_test_score',
        color='mean_test_score',
        labels={'param_n_estimators': 'Number of Estimators', 'param_max_depth': 'Max Depth','param_learning_rate': 'Learning Rate'},
        title='Hyperparameter Tuning Results'
    )

    # Customize the plot layout
    fig.update_layout(
        coloraxis_colorbar=dict(title='Mean Test Score'),
        height=600,
        width=800,
        showlegend=False
    )

    # Show the plot
    fig.show()

    return prediccion_XGB

# XGBoost method for nighttime
def XGB_night(Xtrain,ytrain,Xtest, ytest,col,time):
    

    # Define the hyperparameter distributions
    param_dist = {
        'max_depth': stats.randint(5, 7),
        'learning_rate': stats.uniform(0.02, 0.06),
        'n_estimators':stats.randint(1400, 1700),
        'subsample': stats.uniform(0.5, 0.5),
    }


    # Create the XGBoost model object
    xgb_modeln = xgb.XGBRegressor()

    # Create the RandomizedSearchCV object
    random_searchn = RandomizedSearchCV(xgb_modeln, param_distributions=param_dist, n_iter=10, cv=10, scoring='neg_mean_squared_error',verbose=1,random_state=1234,n_jobs=60)

    # Fit the RandomizedSearchCV object to the training data
    random_searchn.fit(Xtrain, ytrain)

    f = open ('parametersXGB_{time}.txt'.format(time = time),'w')
    f.write("Best set of hyperparameters XGB {time}: {res}".format(time = time, res = random_searchn.best_params_))
    f.close()

    # Use best parameters
    model = random_searchn.best_estimator_

    model.save_model("XGB_SAT_NIGHT_noah.json")

    # Making predictions on the same data or new data
    prediccion_XGB = model.predict(Xtest)

    # Feature importance from the XGBoost model
    importances = model.feature_importances_

    features_sel= col
    indices = np.argsort(importances)

    plt.figure(1, figsize=(10,6))
    plt.title('Feature Importances XGBoost')
    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [features_sel[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.xlim(0, 0.9)
    plt.tight_layout()
    plt.savefig('importance_xgboost_{time}.png'.format(time = time))
    plt.close()
    results = pd.DataFrame(random_searchn.cv_results_)

    fig = px.scatter_3d(
        results,
        x='param_n_estimators',
        y= 'param_max_depth',
        z = 'param_learning_rate',
        color='mean_test_score',
        labels={'param_n_estimators': 'Number of Estimators', 'param_max_depth': 'Max Depth','param_learning_rate': 'Learning Rate'},
        title='Hyperparameter Tuning Results'
    )

    # Customize the plot layout
    fig.update_layout(
        coloraxis_colorbar=dict(title='Mean Test Score'),
        height=600,
        width=800,
        showlegend=False
    )

    # Show the plot
    fig.show()

    return prediccion_XGB

# k-nearest neighbors
def KNN(Xtrain,ytrain,Xtest, ytest,time):

    neighbors = np.arange(1, 10, 1)
    scores = []

    # Running for different K values to know which yields the max accuracy. 
    for k in neighbors:   
        clf = KNeighborsRegressor(n_neighbors = k,  weights = 'uniform', p=2,n_jobs=60)
        clf.fit(Xtrain, ytrain)
        score = cross_val_score(clf, Xtrain, ytrain, cv = 5)
        scores.append(score.mean())

    mse = [1-x for x in scores]
    
    optimal_k = neighbors[mse.index(min(mse))]
    f = open ('optimalk_{time}.txt'.format(time = time),'w')
    f.write("Optimal K: {res}".format(res = optimal_k))
    f.close()

    # Use best parameters
    clf_optimal = KNeighborsRegressor(n_neighbors = optimal_k)

    # Fit model
    clf_optimal.fit(Xtrain, ytrain)

    # Model predictions
    y_pred = clf_optimal.predict(Xtest)

    return y_pred


# Daytime MLP neuronal network
def MLP_day(Xtrain,ytrain,Xtest):

    # Apply a Scaler to input variables
    scaler = RobustScaler()
    scaler = scaler.fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    
    # Select NN model
    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(keras.layers.Dense(18, kernel_initializer='normal',input_dim = Xtrain.shape[1], activation='relu'))

    # The Hidden Layers :
    NN_model.add(keras.layers.Dense(8, kernel_initializer='normal',activation='relu'))
    NN_model.add(keras.layers.Dense(4, kernel_initializer='normal',activation='relu'))
   

    # The Output Layer :
    NN_model.add(keras.layers.Dense(1, kernel_initializer='normal',activation='linear'))

    # Define the learning rate decay
    lr_schedule = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate = 0.01,
    decay_steps = 1000,
    end_learning_rate=0.001,
    power=1,
    cycle=False,
    name="PolynomialDecay",
)
   
    # Select optimizer
    optimizer = keras.optimizers.Adam(learning_rate= lr_schedule)

    # Compile the network :
    NN_model.compile(loss='mse', optimizer= optimizer, metrics=['mse'])
    NN_model.summary()

    # Configure early stop
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20) 

    # Fit model
    hist = NN_model.fit(Xtrain, ytrain, epochs=1000, batch_size=32, validation_split = 0.2, callbacks = early_stop,)

    # Plot model evolution
    def plot_history(history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch


        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error')
        plt.plot(hist['epoch'], hist['mse'],
                label='Train Error')
        plt.plot(hist['epoch'], hist['val_mse'],
                label = 'Val Error')
        plt.ylim([4,20])
        plt.legend()
        plt.savefig('mlp_day.png')
        plt.close()


    plot_history(hist)
 

    # Model predictions
    predictions = NN_model.predict(Xtest)

    return predictions

# Nighttime MLP neuronal network
def MLP_night(Xtrain,ytrain,Xtest):

    # Apply a Scaler to input variables
    scaler = RobustScaler()
    scaler = scaler.fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    
    # Select NN model
    NN_model1 = Sequential()

    # The Input Layer :
    NN_model1.add(keras.layers.Dense(12, kernel_initializer='normal',input_dim = Xtrain.shape[1], activation='relu'))

    # The Hidden Layers :
    NN_model1.add(keras.layers.Dense(8, kernel_initializer='normal',activation='relu'))
    NN_model1.add(keras.layers.Dense(4, kernel_initializer='normal',activation='relu'))

   

    # The Output Layer :
    NN_model1.add(keras.layers.Dense(1, kernel_initializer='normal',activation='linear'))

    # Define the learning rate decay
    lr_schedule = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate = 0.01,
    decay_steps = 1000,
    end_learning_rate=0.001,
    power=1,
    cycle=False,
    name="PolynomialDecay",
    )
    # Select optimizer
    optimizer = keras.optimizers.Adam(learning_rate= lr_schedule)

    # Compile the network :
    NN_model1.compile(loss='mse', optimizer= optimizer, metrics=['mse'])
    NN_model1.summary()

    # Configure early stop
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10) 

    # Fit model
    hist = NN_model1.fit(Xtrain, ytrain, epochs=1000, batch_size=32, validation_split = 0.2, callbacks = early_stop)

    # Plot model evolution
    def plot_history(history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch


        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error ')
        plt.plot(hist['epoch'], hist['mse'],
                label='Train Error')
        plt.plot(hist['epoch'], hist['val_mse'],
                label = 'Val Error')
        plt.ylim([1,15])
        plt.legend()
        plt.savefig('mlp_night.png')
        plt.close()


    plot_history(hist)
 
     # Model predictions
    predictions = NN_model1.predict(Xtest)

 

    return predictions
