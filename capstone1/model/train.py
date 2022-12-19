import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mutual_info_score, make_scorer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import time
import argparse
import logging
import bentoml

FORMAT = "%(asctime)s %(filename)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)

cols_to_keep = [
    "CLIENTNUM",
    "Attrition_Flag",
    "Customer_Age",
    "Dependent_count",
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
    "Months_on_book",
    "Total_Relationship_Count",
    "Credit_Limit",
    "Total_Revolving_Bal"
]

def clean_names(df):
    """Rename some of the columns in the DataFrame"""
    col_rename = {
        "CLIENTNUM": "client_num",
        "Attrition_Flag": "attrition_flag",
        "Customer_Age": "customer_age",
        "Dependent_count": "dependent_count",
        "Gender": "gender",
        "Education_Level": "education_level",
        "Marital_Status": "marital_status",
        "Income_Category": "income_category",
        "Card_Category": "card_category",
        "Months_on_book": "months_on_book",
        "Total_Relationship_Count": "total_relationship_count",
        "Credit_Limit": "credit_limit",
        "Total_Revolving_Bal": "total_revolving_bal"
    }
    df.rename(columns=col_rename, inplace=True)


def split_dataset(df, 
                  size1=0.2, 
                  size2=0.25, 
                  random_state=42
                 ):
    """Split input DataFrame into train, validation and test"""
    train_df, test_x = train_test_split(df, test_size=size1, random_state=random_state)
    train_x, val_x = train_test_split(train_df, test_size=size2, random_state=random_state)
    return train_x.reset_index(drop=True), val_x.reset_index(drop=True), test_x.reset_index(drop=True)

def data_prep(
    df, 
    split,
    oe,
    dv,
    is_ohe=False,
    is_train=True,
    is_drop=True
):
    """Prepare dataset"""
    df = split_dataset(df)[split]

    y = df["attrition_flag"].apply(lambda x: 1 if x == "Attrited Customer" else 0)
    try:
        df = df.drop(["client_num", "attrition_flag"], axis=1)
    except:
        pass
    if is_drop:
        try:
            for col in ["months_on_book", "total_relationship_count", "total_revolving_bal"]:
                del df[col]
        except:
            pass
    if not is_ohe:
        cat_fields = ["education_level", "marital_status", "income_category", "card_category"]
        if is_train:
            df[cat_fields] = oe.fit_transform(df[cat_fields])
        else:
            df[cat_fields] = oe.transform(df[cat_fields])
    dicts = df.to_dict(orient='records')
    if is_train:
        dv.fit(dicts)
    df = dv.transform(dicts)
#     if is_train:
#         scaler.fit(df)
#     df = scaler.transform(df)
    df = pd.DataFrame(df, columns=dv.get_feature_names_out())

    return df, y

def roc_auc_scorer(y_true, y_pred):
    """
    Make scorer to compute roc auc score
    """
    return roc_auc_score(y_true, y_pred)

def build_model(model, train_X, train_y, valid_X, valid_y, 
                hyperparameters={}, 
                scoring=None, cv=3, model_str=None, cat_features=[], verbose=False):
    """
    Build model
    """
    np.random.seed(42)
    best_rmse = 0
    start_time = time.time()
    
    gs = GridSearchCV(model, param_grid=hyperparameters, cv=cv, scoring=scoring)
    gs.fit(train_X, train_y)
    if model_str == 'xgboost':
        gs.best_estimator_.fit(train_X, train_y, verbose=verbose)
#     elif model_str == "rf":
#         gs.best_estimator_.fit(train_X, train_y)
    else:
        gs.best_estimator_.fit(train_X, train_y)
    preds = gs.best_estimator_.predict(valid_X)
    best_roc_auc = roc_auc_score(valid_y, preds)
    end_time = time.time() - start_time
    return gs.best_estimator_, best_roc_auc, np.round(end_time, 2)

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
    # take a subset of the columns
    new_df = df[cols_to_keep].copy()
    logging.info("Clean the field names")
    clean_names(new_df)

    dv = DictVectorizer(sparse=False)
    oe = OrdinalEncoder(encoded_missing_value=-1)

    logging.info("Split and prepare the train, validation and test sets")
    train_x, train_y = data_prep(
        new_df, 
        0,
        oe,
        dv,
        is_drop=False,
        is_ohe=False
    )
    val_x, val_y = data_prep(
        new_df, 
        1, 
        oe,
        dv, 
        is_train=False,
        is_drop=False,
        is_ohe=False
    )
    test_x, test_y = data_prep(
        new_df, 
        2,  
        oe,
        dv,
        is_train=False,
        is_drop=False,
        is_ohe=False
        
    )
    
    logging.info("Train the model")
    lr = LogisticRegression(solver="liblinear", class_weight='balanced', random_state=42)
    best_lr_model, best_lr_rmse, best_lr_time = build_model(lr, train_x, train_y, 
                                                            val_x, val_y,
                                                            hyperparameters={"C": [.01, .1, 1]}, 
                    scoring=make_scorer(roc_auc_scorer, needs_threshold=True), cv=5, model_str="lr")
    logging.info("Training completed")

    logging.info("Pickle the model using Bentoml")
    bentoml.sklearn.save_model(
        'credit_card_churn_model',
        best_lr_model,
        custom_objects=
        {
            "dictVectorizer": dv
        },
        signatures={
          "predict_proba": {
                        "batchable": True,
                        "batch_dim": 0
                        }
        }
    )

if __name__ == "__main__":
    main()
