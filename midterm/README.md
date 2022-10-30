# Car purchase
The objective of this exercise is to build, train and serve a model which predicts how much a customer would pay for a car based on certain customer attributes. The attributes describing a customer are as follows:</br>
   -  `customer name`</br>
   -  `customer email` </br>
   -  `country` </br>
   -  `gender` </br>
   -  `age` </br>
   -  `annual salary` </br>
   -  `credit card debt`</br>
   -  `net worth`</br>
The response variable is: `Car Purchase Amount`

# Model selection
Three models were built and trained with their respective hyperparameters: Linear Regression, RandomForest Regressor and XGB Regressor. Based on the metric - RMSE, Linear Regression was the best performing model of the lot.

# How to run and serve the model
   ```
   git clone https://github.com/sl2902/mlbookcamp-code.git
   ```
   If you would like to only clone the `midterm` subdirectory, then run the following commands. Note: this requires git version >= 2.30.0
   ```
   git clone --depth 1 --filter=blob:none --sparse https://github.com/sl2902/mlbookcamp-code.git. 
   cd mlbookcamp-code
   git sparse-checkout set midterm
   ```
   ```
   cd midterm
   ```
   This should be the structure of the `midterm` subdirectory
   ```
   ├── Pipfile
   ├── Pipfile.lock
   ├── bentofile.yaml
   ├── model
   │   └── train.py
   ├── notebooks
   │   └── Car_purchase_price_prediction.ipynb
   └── predict.py
   ```
   Create a directory called `data` and download the file from (Car purchasing.csv)[https://www.kaggle.com/datasets/yashpaloswal/ann-car-sales-price-prediction?resource=download]
   ```
   mkdir data; cd data
   ```
   Activate the environment
   ```
   pipenv shell
   ```
   Sync the packages from the Pipfile
   ```
   pipenv sync
   ```
   In the `midterm` subdirectory, run the following command to train and save the best model
   ```
   python model/train.py --file-path ./data/car_purchasing.csv
   ```
   Build bentoml
   ```
   bentoml build
   ```
   Build a Docker container using bentoml. Replace the `tag value` with the `tag` from the above command
   ```
   bentoml containerize <<tag value>>
   ```
   Once the Docker image is successfully built, copy the docker command from the standard output, it should look something like this:
   ```
   docker run -it --rm -p 3000:3000 <<tag value>> serve --production
   ```
   Run the above command replacing `tag value`. Once Docker is running, it should provide the following Swagger UI URI which looks like this `http://0.0.0.0:3000`
   Launch the Swagger UI from the browser, and try it out.
