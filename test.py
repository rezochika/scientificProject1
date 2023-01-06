import joblib
import numpy as np

print(joblib.__version__)

model = joblib.load("modelLSTM6420230105/model.pkl")

print(model)
