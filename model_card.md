# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The code in this repository presents a training pipeline for a Binary classification. It uses the Random Forest classifier from the scikit-learn library. It was fitted to learn the classification of a person predicting if his or her income is higher or lower than 50K per year.

The model was trained with the hyper-parameters by default. A grid search could be done to improve the performance of the model.

## Intended Use
The goal of the model is to predict the salary of a person based on some social-economics characteristics.

## Training Data
The training data census.csv is a well-know dataset that is described using pandas profiling [here](https://archive.ics.uci.edu/ml/datasets/census+income).

This dataset has 14 variables: 6 numerical and 8 categorical. The total number of observations are 32561.

## Evaluation Data
The evaluation of the data is done in the test dataset.

## Metrics
Three different metrics have been used in this project:
* Precicion: 0.74
* Recall: 0.64
* F1: 0.68

To better analyze the results of the model, another calculation by category has been also implemented (see `model/metrics_by_slice.csv`). Using this file, we can better understand the behavior of the model for a given value of any categorical variable.

## Ethical Considerations
Given that the raw dataset contains census information from the US only, the trained model could be only applied for american people or residents in the US. Furthermore, this model could have some biases based on sex, race, native-country, and age. Indeed, the dataset has unbalanced categories for race, native-country, and sex variables.

The predictions of the model should be taken carefully according the people on whom this model is used.

## Caveats and Recommendations
The model is limited to the US residents. Furthermore, the raw dataset is unbalanced for some variables. Some solutions to avoid any bias in the trained model is to balance the dataset or to train the model using up sampling and down sampling techniques such as SMOTE.