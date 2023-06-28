import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os


src_path = os.path.dirname(os.path.abspath(__file__)) # Folder where data_processing.py is
data_path = os.path.abspath(os.path.join(src_path,"..","data")) # .. means to go back a folder level from src
df = pd.read_csv(os.path.join(data_path, "raw", "Expanded_data_with_more_features.csv"))

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

df.Gender = lb.fit_transform(df.Gender)

df.LunchType = lb.fit_transform(df.LunchType)

df.dropna(subset=['IsFirstChild'], inplace=True)
df.IsFirstChild = lb.fit_transform(df.IsFirstChild)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df.EthnicGroup = le.fit_transform(df.EthnicGroup)

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

parents = { 'single':1,
            'widowed':1, 
            'divorced':1,
            np.nan:1, 
            'married':2}

df['ParentMaritalStatus'] = df['ParentMaritalStatus'].map(parents)


sport = {'regularly':1,
         'sometimes':2,
         'never':3,
         np.nan:3}

df['PracticeSport'] = df['PracticeSport'].map(sport)
df['PracticeSport'] = df['PracticeSport'].astype(int)

df['NrSiblings'].fillna(0.0, inplace=True)
df['NrSiblings'] = df['NrSiblings'].astype(int)

trans = {'private':2,
         'school_bus':1,
         np.nan:1}

df['TransportMeans'] = df['TransportMeans'].map(trans)

df = df.drop(columns=['Unnamed: 0'])

df2 = df.drop(columns=['ReadingScore', 'WritingScore'])

df2['MathScore'] = df2['MathScore'].apply(lambda x: 1 if x >= 70 else 0)




df2.to_csv(os.path.join(data_path, "processed", 'processed.csv'))

print(df2)
