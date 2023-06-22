import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

src_path = os.path.dirname(os.path.abspath(__file__)) # Folder where data_processing.py is
data_path = os.path.abspath(os.path.join(src_path,"..","data")) # .. means to go back a folder level from src
mod_path = os.path.abspath(os.path.join(src_path,"..","models")) # .. means to go back a folder level from src


df = pd.read_csv(os.path.join(data_path, "processed", "processed.csv"),index_col=0)
#print(df)

X = df[['ParentEduc','TestPrep', 'WklyStudyHours','ReadingScore','WritingScore']]

y = df[['MathScore']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print(X_train.head())

# Create a linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Create a DataFrame for training data
train_data = pd.DataFrame(X_train)
train_data['MathScore'] = y_train

# Save training data to CSV
train_data.to_csv(os.path.join(data_path, "train", 'train.csv'), index=False)

# Create a DataFrame for test data
test_data = pd.DataFrame(X_test)
test_data['MathScore'] = y_test

# Save test data to CSV
test_data.to_csv(os.path.join(data_path, "test", 'test.csv'), index=False)

# Save the trained model
model_file = os.path.join(mod_path, "trained_model.pkl")
with open(model_file, 'wb') as file:
    pickle.dump(model, file)