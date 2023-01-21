from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from .data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    clf = RandomForestClassifier()  
    # Training the model on the training dataset
    # fit function is used to train the model using the training sets as parameters
    rf_model = clf.fit(X_train, y_train)
    
    return rf_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # performing predictions on the test dataset
    y_pred = model.predict(X)
    return y_pred


def compute_metrics_by_slice(clf, encoder, lb, df, target, cat_columns, output_path=None):
    """Compute metrics by slice of the data.
    For simplicity, the function outputs the performance on slices of just the categorical features.
    Inputs
    ------
    clf: Classifier model (scikit-learn compliant)
    encoder: trained one-hot-encoder (output of the data.process_data function on training data)
    lb: trained label binarizer (output of the data.process_data function on training data)
    df: pandas dataframe where to compute metrics by slice
    target: target column in the df input dataframe
    cat_columns: categorical columns
    output_path: output path to write the output dataframe.
    Returns
    -------
    metrics_df: pd.DataFrame
        Predictions by slice on the input data according the categorical columns.
    """
    columns = ["column", "category", "precision", "recall", "f1"]
    metrics_df = pd.DataFrame(columns=columns)

    for col in cat_columns:
        for category in df[col].unique():

            df_line = {}

            tmp_df = df[df[col] == category]

            X, y, _, _ = process_data(
                X=tmp_df,
                categorical_features=cat_columns,
                label=target,
                training=False,
                encoder=encoder,
                lb=lb
            )

            preds = inference(clf, X)

            precision, recall, f1 = compute_model_metrics(y, preds)

            df_line['column'] = col
            df_line['category'] = category
            df_line['precision'] = precision
            df_line['recall'] = recall
            df_line['f1'] = f1

            metrics_df = metrics_df.append(df_line, ignore_index=True)

    if output_path is not None:
        metrics_df.to_csv(output_path)

    return metrics_df