import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score

import pickle

src_path = os.path.dirname(os.path.abspath(__file__)) # Folder where data_processing.py is
data_path = os.path.abspath(os.path.join(src_path,"..","data")) # .. means to go back a folder level from src
mod_path = os.path.abspath(os.path.join(src_path,"..","models")) # .. means to go back a folder level from src


df = pd.read_csv(os.path.join(data_path, "processed", "processed.csv"),index_col=0)
#print(df)

X = df.drop('MathScore', axis=1)
y = df['MathScore']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Preprocessing step: StandardScaler
    ('pca', PCA(n_components=11)),  # Preprocessing step: PCA
    ('model', LogisticRegression())  # Best model: LogisticRegression
])

# Fit the pipeline to the training data
pipeline.fit(X_train_resampled, y_train_resampled)


# Create a DataFrame for training data
train_data = pd.DataFrame(X_train_resampled)
train_data['MathScore'] = y_train_resampled

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
    pickle.dump(pipeline, file)