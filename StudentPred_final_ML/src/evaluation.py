import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

src_path = os.path.dirname(os.path.abspath(__file__)) # Folder where data_processing.py is
data_path = os.path.abspath(os.path.join(src_path,"..","data")) # .. means to go back a folder level from src
mod_path = os.path.abspath(os.path.join(src_path,"..","models")) # .. means to go back a folder level from src

model_file = os.path.join(mod_path, "trained_model.pkl")

with open(model_file, 'rb') as file:
    loaded_model = pickle.load(file)

#print(loaded_model)

df = pd.read_csv(os.path.join(data_path, "test", "test.csv"))
print(df.head())

X_test = df[['ParentEduc','TestPrep', 'WklyStudyHours','ReadingScore','WritingScore']]
y_test = df[['MathScore']]

#print(X_test.columns)

y_pred = loaded_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the predicted values alongside the actual test values
#results = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
#print(results)

# Display evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("\nMean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)