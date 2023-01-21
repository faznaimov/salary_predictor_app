# Script to train machine learning model.
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, compute_metrics_by_slice

# Load the data.
data = pd.read_csv("./data/census.csv")
data.columns = [col.strip() for col in data.columns]

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.2, stratify=data['salary'], random_state=15)

cat_features = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'education-num'
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label='salary', training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label='salary', training=True, encoder=encoder, lb=lb
)

#print(X_train.shape)
#print(X_test.shape)

# Train a model.
model = train_model(X_train, y_train)

# Save the model.
joblib.dump(model, "./model/model.pkl")

# Get preditions on test data.
y_pred = inference(model, X_test)

# Get metrics on test data.
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
print(f"Precicion: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1: {fbeta:.2f}")

compute_metrics_by_slice(
    clf=model,
    encoder=encoder,
    lb=lb,
    df=test,
    target='salary',
    cat_columns=cat_features,
    output_path="model/metrics_by_slice.csv",
)
