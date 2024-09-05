# Voter Turnout Prediction using GBM Regressor

Overview

This repository contains code to predict voter turnout using voter file data from Oklahoma. The model utilizes the sklearn Histogram-based Gradient Boosting Regression Tree (HistGradientBoostingRegressor) ensemble method, chosen for its efficiency with large datasets (approximately 400,000 records). This model helps in understanding voter behavior and can be instrumental in strategizing outreach and engagement efforts.


Why Use HistGradientBoostingRegressor?

The HistGradientBoostingRegressor is well-suited for this task because it handles large datasets effectively, unlike the standard Gradient Boosting Regressor in sklearn, which is limited to datasets with fewer than 10,000 records.


Data

The model is trained on a voter file dataset containing the following fields:
vote history (y or target variable), 
age, 
years of registration,
registered party, 
and race


Installation 

-sklearn
-matplotlib
-numpy
-pandas
