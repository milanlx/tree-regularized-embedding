import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score


# hyperparameter
xgb_params = {
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'tree_method': 'approx', 
    'verbosity': '2'
}

xgb_option = {
    'num_boost_round': 500,
    'early_stopping_round': 10
}


def convert_to_DMatrix(X_data, Y_data):
    return xgb.DMatrix(data=X_data, label=Y_data.astype(int))


def train_model(train_data, valid_data, xgb_params, xgb_option):
    xgb_model = xgb.train(
        params=xgb_params, 
        dtrain=train_data, 
        evals=[(valid_data, 'valid')],
        early_stopping_rounds=xgb_option['early_stopping_round'],
        num_boost_round=xgb_option['num_boost_round']
    )
    return xgb_model


def evaluate_model(xgb_model, test_data):
    y_pred = xgb_model.predict(test_data)
    y_true = test_data.get_label()
    auc = roc_auc_score(y_true, y_pred) 
    return auc 