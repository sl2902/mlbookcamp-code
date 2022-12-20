# Credit card churn predictor
The objective of this exercise is to build, train and serve a model which predicts whether a bank's customer is likely to churn based on their credit card history. The attributes describing a customer are as follows:</br>
- `CLIENTNUM`  - Unique identifier for the customer holding the account<br/>
- `Attrition_Flag`   - Internal event (customer activity) variable - if the account is closed then 1 else 0<br/>
- `Customer_Age`  - Demographic variable - Customer's Age in Years<br/>
- `Gender`  - Demographic variable - M=Male, F=Female<br/>
- `Dependent_count`  - Demographic variable - Number of dependents<br/>
- `Education_Level`  - Demographic variable - Educational Qualification of the account holder (example: high school, college graduate, etc.)<br/>
- `Marital_Status`   - Demographic variable - Married, Single, Divorced, Unknown<br/>
- `Income_Category`  - Demographic variable - Annual Income Category of the account holder (< $40K, $40K - 60K, $60K - $80K, $80K-$120K, ><br/>
- `Card_Category` - Product Variable - Type of Card (Blue, Silver, Gold, Platinum)<br/>
- `Monthsonbook`  - Period of relationship with bank<br/>
- `TotalRelationshipcount` - Total no. of products held by the customer<br/>
- `MonthsInactive12_mon`   - No. of months inactive in the last 12 months<br/>
- `ContactsCount12_mon` - No. of Contacts in the last 12 months<br/>
- `Credit_Limit`  - Credit Limit on the Credit Card<br/>
- `TotalRevolvingBal`   - Total Revolving Balance on the Credit Card<br/>
- `AvgOpenTo_Buy` - Open to Buy Credit Line (Average of last 12 months)<br/>
- `TotalAmtChngQ4Q1` - Change in Transaction Amount (Q4 over Q1)<br/>
- `TotalTransAmt` - Total Transaction Amount (Last 12 months)<br/>
- `TotalTransCt`  - Total Transaction Count (Last 12 months)<br/>
- `TotalCtChngQ4Q1`  - Change in Transaction Count (Q4 over Q1)<br/>
- `AvgUtilizationRatio` - Average Card Utilization Ratio<br/>
The response variable is: `Attrition flag`

# Model selection
Three models were built and trained with their respective hyperparameters: Logistic Regression, RandomForest Classifier and XGB Classifier. Based on the metric - ROC AUC, Logistic Regression was the best performing model of the lot.

# How to run and serve the model
   ```
   git clone https://github.com/sl2902/mlbookcamp-code.git
   ```
   If you would like to only clone the `capstone1` subdirectory, then run the following commands. Note: this requires git version >= 2.30.0
   ```
   git clone --depth 1 --filter=blob:none --sparse https://github.com/sl2902/mlbookcamp-code.git;
   cd mlbookcamp-code;
   git sparse-checkout set capstone1
   ```
   Switch to subdirectory `capstone1`
   ```
   cd capstone1
   ```
   This should be the structure of the `capstone1` subdirectory
   ```
   ├── Pipfile
   ├── Pipfile.lock
   ├── README.md
   ├── bentofile.yaml
   ├── model
   │   └── train.py
   ├── notebooks
   │   └── Credit_card_churn.ipynb
   └── predict.py
   ```
   Create a directory called `data` and download the file from [BankChurners.csv](https://www.kaggle.com/datasets/whenamancodes/credit-card-customers-prediction?resource=download)
   ```
   mkdir data; cd data
   ```
   Back in the `capstone1` subdirectory, activate the environment
   ```
   pipenv shell
   ```
   Sync the packages from the Pipfile
   ```
   pipenv sync
   ```
   In the `captone1` subdirectory, run the following command to train and save the best model
   ```
   python model/train.py --file-path ./data/BankChurners.csv
   ```
   Build bentoml
   ```
   bentoml build
   ```
   Make sure you have Docker installed, and that it is running. Build a Docker container using bentoml. Replace the `tag value` with the successfully built model `tag` from the above command
   ```
   bentoml containerize <<tag value>>
   ```
   Output from the above command, which also shows the command for the next steps below
   
   <img width="1786" alt="image" src="https://user-images.githubusercontent.com/7212518/208729983-23730ab3-af6e-405e-99a6-259659a8613a.png">

   Once the Docker image is successfully built, copy the docker command from the standard output, it should look something like this:
   ```
   docker run -it --rm -p 3000:3000 <<tag value>> serve --production
   ```
  
   Run the above command making sure `tag value` contains the latest id. Once Docker is running, it should provide the following Swagger UI URI which looks like this `http://0.0.0.0:3000`. </br>

   The schema for the POST API is as follows:
- customer_age: int
- dependent_count: int
- gender: str
- education_level: str
- marital_status: str 
- income_category: str 
- card_category: str 
- credit_limit: int
- months_on_book: int 
- total_relationship_count: int
- total_revolving_bal: float

For the `str` fields, the following are all the valid values that the respective fields take:
- `gender` - M, F
- `education_level` - Graduate, High School, Unknown, Uneducated, College, Post-Graduate, Doctorate
- `marital_status` - Married, Single, Unknown, Divorced
- `income_category` - Less than $40K, $40K - $60K, $80K - $120K, $60K - $80K, Unknown, $120K +
- `card_category` - Blue, Silver, Gold, Platinum

Example you can play with
```
{
  "customer_age": 45,
  "dependent_count": 3,
  "gender": "M",
  "education_level": "Graduate",
  "marital_status": "Single",
  "income_category": "$120K +",
  "card_category": "Blue",
  "credit_limit": 2400,
  "months_on_book": 0,
  "total_relationship_count": 1,
  "total_revolving_bal": 1343
}
```


   Launch the Swagger UI from the browser, and try it out. Sample output
   <img width="1680" alt="image" src="https://user-images.githubusercontent.com/7212518/208729168-f8038a49-8cc3-40db-ae15-89620e427b20.png">

<img width="1424" alt="image" src="https://user-images.githubusercontent.com/7212518/208729251-00fa9c9e-b565-4a0b-8a72-da5808a89787.png">

