import json
import requests

data = {
    "age": 35,
    "workclass": "State-gov",
    "fnlgt": 141297,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "Asian-Pac-Islander",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

response = requests.post('https://salary-predictor.onrender.com/predictions', data=json.dumps(data))

print(response.status_code)
print(response.json())