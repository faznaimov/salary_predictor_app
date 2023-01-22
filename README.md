# Salary Predictor Application with FastAPI

- Project **Deploying a ML Model to Cloud Application Platform with FastAPI** in [ML DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821)  program by Udacity.

## Table of Contents

- [Introduction](#ml-pipeline-for-short-term-rental-prices-in-nyc)
- [Project Description](#project-description)
- [Files and Data Description](#files-and-data-description)
- [Usage](#usage)
  * [Create Environment](#create-environment)
  * [Run The App on Local Machine](#run-the-app-on-local-machine)
  * [Run Test Scripts](#run-test-scripts)
- [License](#license)

## Project Description
Salary prediction model that is trained on [Census Bureau data](https://archive.ics.uci.edu/ml/datasets/census+income) and deployed with FastAPI. A remote DVC pointing to AWS S3 bucket that tracks data changes. In addition, wrote unit tests to monitor the model performance on various slices of the data. Deployed my model using the FastAPI package and created API tests. Both the slice-validation and the API tests are incorporated into a CI/CD framework using GitHub Actions.

[Deployed App](https://salary-predictor.onrender.com)

## App screenshot

![App](/screenshots/app.png)

## Files and Data description
The directories structure are list as below:
```bash
.
├── data
│   └── census.csv.dvc
├── model
│   ├── metrics_by_slice.csv
│   └── model.pkl
├── screenshots
│   └── app.png
├── starter
│   ├── ml
│   │    ├── data.py
│   │    └── model.py
│   ├── config.py
│   └── train_model.py
├── tests
│   ├── test_main.py
│   └── test_model.py
├── README.md
├── main.py
├── model_card.md
└── requirements.txt
```


- ```census.csv.dvc```: DVC info of the dataset that is located in AWS S3 bucket
- ```metrics_by_slice.csv```: Detailed model metrics on categorical data
- ```model.pkl```: Random Forest model
- ```data.py```: Module containing preprocessing function
- ```model.py```: Module containing training, metrics and inference functions
- ```config.py```: Config file for train_model.py
- ```train_model.py```: Script to train model
- ```test_main.py```: Test script for main.py
- ```test_model.py```: Test script for model.py
- ```main.py```: FastAPI app
- ```model_card.md```: Model Card

## Usage

### Create Environment
Make sure to have conda installed and ready.

```bash
> conda create -n [envname] "python=3.8" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
```

### Run The App on Local Machine
```
uvicorn main:app --reload
```

### Run Test Scripts
```
python -m pytest -vv
```

## License

[License](LICENSE.txt)