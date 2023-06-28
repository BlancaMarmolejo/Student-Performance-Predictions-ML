import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score

src_path = os.path.dirname(os.path.abspath(__file__)) # Folder where data_processing.py is
data_path = os.path.abspath(os.path.join(src_path,"..","data")) # .. means to go back a folder level from src
mod_path = os.path.abspath(os.path.join(src_path,"..","models")) # .. means to go back a folder level from src

model_file = os.path.join(mod_path, "trained_model.pkl")

with open(model_file, 'rb') as file:
    loaded_model = pickle.load(file)

#print(loaded_model)

df = pd.read_csv(os.path.join(data_path, "test", "test.csv"))
print(df.head())

X_test = df.drop('MathScore', axis=1)
y_test = df[['MathScore']]


# Make predictions on the test data
y_pred = loaded_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Get the predicted probabilities for the positive class from the pipeline
y_pred_proba = loaded_model.predict_proba(X_test)[:, 1]

# Calculate the AUC score
auc = roc_auc_score(y_test, y_pred_proba)

print("AUC:", auc)

precision = precision_score(y_test, y_pred)
print("Precision:", precision)

recall = recall_score(y_test, y_pred)
print("Recall:", recall)