# Script to train machine learning model.
from ml.model import train_model, compute_model_metrics, inference, compute_metrics_by_slice
from ml.data import process_data
import config
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# Load the data.
data = pd.read_csv(config.data_pth)
data.columns = [col.strip() for col in data.columns]

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(
    data, test_size=config.test_size, stratify=data[config.y], random_state=15)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=config.cat_features, label=config.y, training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=config.cat_features, label=config.y, training=True, encoder=encoder, lb=lb
)

# Train a model.
model = train_model(X_train, y_train)

# Save the model.
joblib.dump(model, config.model_pth)

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
    target=config.y,
    cat_columns=config.cat_features,
    output_path=config.metrics_pth,
)
