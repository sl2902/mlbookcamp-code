import pandas as pd
import numpy as np
import os, sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mutual_info_score, make_scorer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
import time
import argparse
import logging
import bentoml

FORMAT = "%(asctime)s %(filename)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)

def clean_names(df):
    """Rename some of the columns in the DataFrame"""
    col_rename = {
        "customer name": "customer_name",
        "customer e-mail": "email",
        "annual Salary": "annual_salary",
        "credit card debt": "credit_card_debt",
        "net worth": "net_worth",
        "car purchase amount": "car_purchase_amount"
    }
    df.rename(columns=col_rename, inplace=True)


def split_dataset(df, 
                  size1=0.1, 
                  size2=0.11, 
                  random_state=42
                 ):
    """Split input DataFrame into train, validation and test"""
    train_df, test_x = train_test_split(df, test_size=size1, random_state=random_state)
    train_x, val_x = train_test_split(train_df, test_size=size2, random_state=random_state)
    return train_x.reset_index(drop=True), val_x.reset_index(drop=True), test_x.reset_index(drop=True)

def data_prep(df, 
              split,
              oe,
              dv,
              is_gender=True,
              is_country=True, 
              is_train=True):
    """Prepare dataset"""
    df = split_dataset(df)[split]
    try:
        df = df.drop(["customer_name", "email"], axis=1)
    except:
        pass
    if is_gender:
        df["gender"] = df["gender"].apply(lambda x: "male" if x == 1 else "female")
    else:
        del df["gender"]
    if is_country:
        if is_train:
            oe.fit(df["country"].values.reshape(-1, 1))
        df["country"] = oe.transform(df["country"].values.reshape(-1, 1))
    else:
        del df["country"]
    dicts = df.to_dict(orient='records')
    if is_train:
        dv.fit(dicts)
    df = pd.DataFrame(dv.transform(dicts), columns=dv.get_feature_names_out())
    y = df["car_purchase_amount"]
    del df["car_purchase_amount"]
    return df, y

def rmse(y_true, y_pred):
    """Make scorer to compute rmse"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def build_model(model, train_X, train_y, valid_X, valid_y, 
                hyperparameters={}, 
                scoring=None, cv=3, model_str=None, cat_features=[], verbose=False):
    """Train model"""
    np.random.seed(42)
    best_rmse = 0
    start_time = time.time()
    
    gs = GridSearchCV(model, param_grid=hyperparameters, cv=cv, scoring=scoring)
    gs.fit(train_X, train_y)
    if model_str == 'xgboost':
        gs.best_estimator_.fit(train_X, train_y, verbose=verbose)
    elif model_str == "rf":
        gs.best_estimator_.fit(train_X, train_y)
    else:
        gs.best_estimator_.fit(train_X, train_y)
    preds = gs.best_estimator_.predict(valid_X)
    best_rmse = rmse(valid_y, preds)
    end_time = time.time() - start_time
    return gs.best_estimator_, best_rmse, np.round(end_time, 2)

def test_prediction(model, test_X, test_y):
    """Predict RMSE on the test set"""
    start_time = time.time()
    preds = model.predict(test_X)
    return np.round(time.time(), 2) - np.round(start_time, 2), rmse(test_y, preds)

def main():
    parser = argparse.ArgumentParser(description="Train best model")
    parser.add_argument(
        "--file-path", "-f", 
        required=True,
        help="Enter file path and name of file to train"
        )
    args = parser.parse_args()
    file_path = args.file_path
    logging.info("Read input file")
    df = pd.read_csv(file_path, sep=",", encoding="latin1")
    logging.info("Clean the field names")
    clean_names(df)

    dv = DictVectorizer(sparse=False)
    oe = OrdinalEncoder(encoded_missing_value=-1)

    logging.info("Split and prepare the train, validation and test sets")
    train_x, train_y = data_prep(
        df, 
        0, 
        oe,
        dv, 
        is_gender=False,
        is_country=False
        )
    val_x, val_y = data_prep(
        df, 
        1, 
        oe,
        dv,
        is_gender=False,
        is_country=False
        )
    test_x, test_y = data_prep(
        df, 
        2, 
        oe,
        dv,
        is_gender=False,
        is_country=False
        )
    
    logging.info("Train the model")
    lr = LinearRegression()
    best_lr_model, best_lr_rmse, best_lr_time = build_model(lr, train_x, train_y, 
                                                            val_x, val_y,
                                                            hyperparameters={}, 
                    scoring=make_scorer(rmse), cv=5, model_str="lr")
    logging.info("Training completed")

    logging.info("Pickle the model using Bentoml")
    bentoml.sklearn.save_model(
        'car_purchase_model',
        best_lr_model
        )

if __name__ == "__main__":
    main()
