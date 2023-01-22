'''
Module to test main.py

Author: Faz Naimov
Date: 1/21/2023
'''

import pytest
from main import app
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    client_tc = TestClient(app)
    return client_tc

def test_get_success(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {'message': 'Welcome to the salary predictor!'}

def test_post_data_above(client):
    req = {
          "age": 31,
          "workclass": "Private",
          "fnlgt": 45781,
          "education": "Masters",
          "education_num": 14,
          "marital_status": "Never-married",
          "occupation": "Prof-specialty",
          "relationship": "Not-in-family",
          "race": "White",
          "sex": "Female",
          "capital_gain": 14084,
          "capital_loss": 0,
          "hours_per_week": 50,
          "native_country": "United-States"
    }
    r = client.post("/predictions", json=req)
    assert r.status_code == 200
    assert r.json() == {'Predicted salary': ' >50K'}


def test_post_data_below(client):
    req = {
          "age": 32,
          "workclass": "Private",
          "fnlgt": 186824,
          "education": "HS-grad",
          "education_num": 9,
          "marital_status": "Never-married",
          "occupation": "Machine-op-inspct",
          "relationship": "Unmarried",
          "race": "White",
          "sex": "Male",
          "capital_gain": 0,
          "capital_loss": 0,
          "hours_per_week": 40,
          "native_country": "United-States"
    }
    r = client.post("/predictions", json=req)
    assert r.status_code == 200
    assert r.json() == {'Predicted salary': ' <=50K'}