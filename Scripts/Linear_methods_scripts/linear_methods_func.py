import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LinearRegression,Ridge
from sklearn.model_selection import RepeatedKFold
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import AlphaSelection,ManualAlphaSelection
from sklearn.metrics import mean_squared_error, r2_score


# Script with the linear methods used in Vedrí et al. (2025)

# Mlr function is used to apply classic ordinary least squares (OLS) method which is non regularized
def mlr (Xtrain,ytrain,Xtest, ytest):

    # Apply the algorithm from scikit-learn library
    mlr = LinearRegression()  
    mlr.fit(Xtrain, ytrain)    
    prediccion_mlr= mlr.predict(Xtest)    
    
    # Obtain model statistics
    mse_mlr = mean_squared_error(ytest, prediccion_mlr,squared=False)
    r2_mlr = r2_score(ytest, prediccion_mlr)

    # Residual plot
    visualizer = ResidualsPlot(mlr)

    # Fit the training data to the visualizer
    visualizer.fit(Xtrain, ytrain)  

    # Evaluate the model on the test data
    visualizer.score(Xtest, ytest) 

    # Finalize and render the figure 
    visualizer.show()                 

    print(f"Test RMSE MLR: {mse_mlr}")
    print(f"Test R2 MLR: {r2_mlr}")

    coeficient = pd.DataFrame()
    coeficient['variables'] = mlr.feature_names_in_
    coeficient['coefs'] = np.reshape(mlr.coef_,(len(mlr.feature_names_in_),1))
   
    intercept = pd.DataFrame()
    intercept['variables'] = ['intercept']
    intercept['coefs'] = mlr.intercept_

    coeficient = pd.concat([coeficient, intercept]).reset_index(drop=True)

    return prediccion_mlr, coeficient

# Ridge is a regularized method which uses all variables without dropping them
def ridge(Xtrain,ytrain,Xtest, ytest):

    # Applying a 10 kfold
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)

    # Setting an alpha range to optimize the algorithm (actual values were set for our data after several modifications)
    alphas=np.arange(0.001, 1, 0.001)

    # Model creation
    model_ridge = RidgeCV(alphas= alphas,cv = cv, scoring='neg_root_mean_squared_error')

    # Fit model
    model_ridge.fit(Xtrain, ytrain)

    # Summarize chosen configuration
    print('alpha: %f' % model_ridge.alpha_)

    # Model prediction
    prediccion_ridge = model_ridge.predict(Xtest)

    # Instantiate the visualizer
    visualizer = ManualAlphaSelection(
        Ridge(random_state=1234),
        alphas=alphas,
        cv=10,
        scoring="neg_root_mean_squared_error"
    )

    visualizer.fit(Xtrain, ytrain)
    visualizer.show()


    visualizer = ResidualsPlot(model_ridge)

    # Fit the training data to the visualizer
    visualizer.fit(Xtrain, ytrain)
    
    # Evaluate the model on the test data
    visualizer.score(Xtest, ytest)  

    # Finalize and render the figure
    visualizer.show()                 


    #Statistical report

    mse_ridge = mean_squared_error(ytest, prediccion_ridge,squared=False)
    r2_ridge = r2_score(ytest, prediccion_ridge)

    print(f"Test RMSE RIDGE: {mse_ridge}")
    print(f"Test R2 RIDGE: {r2_ridge}")


    return prediccion_ridge

# Lasso is a regularized method which uses all variables being able to drop them
def lasso(Xtrain,ytrain,Xtest, ytest):

    # Applying a 10 kfold
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)

    # Define model to optimize the algorithm (actual values were set for our data after several modifications)
    model_lasso = LassoCV(alphas=np.arange(0.001, 1, 0.001),cv=cv, max_iter = 5000, random_state= 1234,n_alphas=1000)
    
    # Fit model
    model_lasso.fit(Xtrain, ytrain)

    # Summarize chosen configuration
    print('alpha: %f' % model_lasso.alpha_)

    prediccion_lasso = model_lasso.predict(Xtest)

    #Classification report
    visualizer = AlphaSelection(model_lasso)
    visualizer.fit(Xtrain, ytrain)
    visualizer.show()


    mse_lasso = mean_squared_error(ytest, prediccion_lasso,squared=False)
    r2_lasso = r2_score(ytest, prediccion_lasso)

    print(f"Test RMSE LASSO: {mse_lasso}")
    print(f"Test R2 LASSO: {r2_lasso}")
    return prediccion_lasso

# Elastic Net is a regularized method which mix Ridge and Lasso properties
def EN(Xtrain,ytrain,Xtest, ytest):
    
    # Applying a 10 kfold
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)

    # Define model to optimize the algorithm (actual values were set for our data after several modifications)
    model_en = ElasticNetCV(l1_ratio = np.arange(0.001, 1, 0.001),cv = cv ,alphas=np.arange(0.001, 1, 0.001),max_iter = 5000,random_state= 1234,n_alphas=100,verbose=4)

    # Fit model
    model_en.fit(Xtrain, ytrain)

    # Summarize chosen configuration
    print('alpha: %f' % model_en.alpha_)
    print('l1: %f' % model_en.l1_ratio_)

    prediccion_en = model_en.predict(Xtest)

    visualizer = AlphaSelection(model_en)
    visualizer.fit(Xtrain, ytrain)
    visualizer.show()
    #Classification report


    mse_en = mean_squared_error(ytest, prediccion_en,squared=False)
    r2_en = r2_score(ytest, prediccion_en)


    print(f"Test RMSE EN: {mse_en}")
    print(f"Test R2 EN: {r2_en}")

    return prediccion_en



