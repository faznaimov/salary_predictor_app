from fastapi import FastAPI
import pandas as pd
import pickle as pkl
import os
from pydantic import BaseModel, Field

import starter.config as config
from starter.ml.data import process_data
from starter.ml.model import inference


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI(
    title="API for salary predictor",
    description="This API helps to classify",
    version="1.0.0",
)

def hyphen_to_underscore(field_name):
    return f"{field_name}".replace("_", "-")

class InputData(BaseModel):
    age: int = Field(..., example=45)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=2334)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    marital_status: str = Field(..., example="Never-married")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Wife")
    race: str = Field(..., example="Black")
    sex: str = Field(..., example="Female")
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=60)
    native_country: str = Field(..., example="Cuba")
    
    class Config:
        alias_generator = hyphen_to_underscore
        allow_population_by_field_name = True

'''
@app.on_event("startup")
async def startup_event(): 
    global model, encoder, lb
    with open(config.model_pth, 'rb') as f:
        encoder, lb, model = pkl.load(f)
'''

@app.get("/")
async def welcome():
    return {'message': 'Welcome to the salary predictor!'}

@app.post("/predictions")
async def prediction(input_data: InputData):
    
    with open(config.model_pth, 'rb') as f:
        encoder, lb, model = pkl.load(f)
        
    # Formatting input_data
    input_df = pd.DataFrame(
        {k: v for k, v in input_data.dict().items()}, index=[0]
    )
    input_df.columns = [_.replace('_', '-') for _ in input_df.columns]

    # Processing input data
    X, _, _, _ = process_data(
        X=input_df,
        label=None,
        training=False,
        categorical_features=config.cat_features,
        encoder=encoder,
        lb=lb,
    )

    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]

    return {"Predicted salary": y}