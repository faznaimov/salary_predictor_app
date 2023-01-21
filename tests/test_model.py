'''
Module to test model.py

Author: Faz Naimov
Date: 1/21/2023
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import pytest
import os
import starter.ml.model as cls
from starter.ml.data import process_data
import starter.config as config

@pytest.fixture()
def input():
    '''
    pytest fixture that hold reusable data
    '''
    
    df = pd.read_csv(config.data_pth)
    df.columns = [col.strip() for col in df.columns]
    train, test = train_test_split(df, test_size=config.test_size, stratify=df[config.y], random_state=15)
    
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=config.cat_features, label=config.y, training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=config.cat_features, label=config.y, training=True, encoder=encoder, lb=lb
    )

    model = cls.train_model(X_train, y_train)
    
    return test, X_test, y_test, encoder, lb, model

def inference(input):
    '''
    test inference function
    '''
    
    _, X_test, _, _, _, model = input
    
    y_preds = cls.inference(model, X_test)
    
    # All predictions should be less than 1
    assert all(y_preds <= 1.0)

    # All predictions should be positive
    assert all(y_preds >= 0)

def test_compute_model_metrics(input):
    '''
    test compute_model_metrics function
    '''
    
    _, X_test, y_test, _, _, model = input
    
    y_preds = cls.inference(model, X_test)
    precision, recall, fbeta = cls.compute_model_metrics(y_test, y_preds)
    
    # Checking type
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    
    # All metrics should not be higher then 1
    assert precision <= 1.0
    assert recall <= 1.0
    assert fbeta <= 1.0

    # Only for precision, it should be higher than 0.5
    # the hazard limit for a binary classification
    assert precision >= 0.5

def test_compute_metrics_by_slice(input):
    '''
    test compute_metrics_by_slice function
    '''
    
    test, _, _, encoder, lb, model = input
    
    cls.compute_metrics_by_slice(
        clf=model,
        encoder=encoder,
        lb=lb,
        df=test,
        target=config.y,
        cat_columns=config.cat_features,
        output_path=config.metrics_pth,
    )
    
    assert os.path.isfile(config.metrics_pth)