# Car purchase
The objective of this exercise is to build, train and serve a model which predicts how much a customer would pay for a car based on certain customer attributes. The attributes describing a customer are as follows:</br>
    `customer name`</br>
    `customer email` </br>
    `country` </br>
    `gender` </br>
    `age` </br>
    `annual salary` </br>
    `credit card debt`</br>
    `net worth`</br>
The response variable is: `Car Purchase Amount`

# Model selection
Three models were built and trained with their respective hyperparameters: Linear Regression, RandomForest Regressor and XGB Regressor. Based on the metric - RMSE, Linear Regression was the best performing model of the lot.

# How to run and serve the model
git clone the repo. If you would like to only clone the midterm subdirectory, then run the following commands
Note: this requires git version >= 2.30.0
git clone --depth 1 --filter=blob:none --sparse https://github.com/sl2902/mlbookcamp-code.git</br>
cd mlbookcamp-code</br>
git sparse-checkout set midterm
