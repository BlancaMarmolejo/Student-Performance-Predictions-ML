import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os


src_path = os.path.dirname(os.path.abspath(__file__)) # Folder where data_processing.py is
data_path = os.path.abspath(os.path.join(src_path,"..","data")) # .. means to go back a folder level from src
df = pd.read_csv(os.path.join(data_path, "raw", "Expanded_data_with_more_features.csv"))

mapping = {
    '< 5': 1,
    '5 - 10': 2,
    '> 10': 3,
    float('nan'): 2
}

df['WklyStudyHours'] = df['WklyStudyHours'].map(mapping)

mappingp = {
    "some high school": 1,
    "high school": 2,
    "some college": 3,
    "associate's degree": 4,
    "bachelor's degree": 5,
    "master's degree": 6,
    "doctoral degree": 7,
    float('nan'): 3
}

df['ParentEduc'] = df['ParentEduc'].map(mappingp)

mapping = {
    "none": 1,
    "completed": 2,
    float('nan'): 1
}

df['TestPrep'] = df['TestPrep'].map(mapping)

df2 = df[['ParentEduc','TestPrep', 'WklyStudyHours','ReadingScore','WritingScore','MathScore']]

df2.to_csv(os.path.join(data_path, "processed", 'processed.csv'))

print(df2)
