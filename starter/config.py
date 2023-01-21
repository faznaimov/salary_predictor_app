data_pth = "./data/census.csv"
test_size = 0.2
y = 'salary'
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
model_pth = "./model/model.pkl"
metrics_pth = "model/metrics_by_slice.csv"