# Import libraries

from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import subprocess
import os

# Title
st.set_page_config(page_title = 'Predicitng Scores', page_icon= ":book:")
st.title('Predicting Student Math Scores:')
st.header('Machine Learning Classification')
st.image('https://img.freepik.com/premium-vector/realistic-math-chalkboard-background_23-2148154055.jpg',
         caption='Blanca T. Marmolejo')

st.divider()

st.markdown('Our goal is early detection of students who may be at risk of performing poorly on the end of year Math exam.')
st.markdown('We use data such as student characteristics, parental factors, and environmental factors to offer support to students in need of additional assistance.')

st.divider()

# Datasets


src_path = os.path.dirname(os.path.abspath(__file__)) # Folder where data_processing.py is
data_path = os.path.abspath(os.path.join(src_path,"..","data")) # .. means to go back a folder level from src
df = pd.read_csv(os.path.join(data_path, "raw", "Expanded_data_with_more_features.csv"))

mod_path = os.path.abspath(os.path.join(src_path,"..","models")) # .. means to go back a folder level from src
df2 = pd.read_csv(os.path.join(data_path, "processed", "processed.csv"),index_col=0)

st.header('Raw data')

st.dataframe(df)

st.divider()

st.header("Preprocessing")

st.code("""
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

""")

st.divider()

st.header("Clean Data")

st.dataframe(df2)

st.divider()



st.header('Balancing data and train best model')

st.code("""
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
""")

st.divider()


st.header('Applied a variety of models including:')
st.markdown('Pipeline above with Logistic Regression        AUC: 0.743')
st.markdown('Voting Classifier      AUC: 0.735')
st.markdown('Support Vector Machine     AUC: 0.734')
st.markdown('Gradient Boosting      AUC: 0.726')
st.markdown('Decision Tree      AUC: 0.716')

st.divider()


st.header('Best Model')

st.markdown("We use the first pipeline model listed above")

model_file = os.path.join(mod_path, "trained_model.pkl")

with open(model_file, 'rb') as file:
    loaded_model = pickle.load(file)

st.header('Input desired parameters:')

Gender = st.slider('gender', 0, 1)
EthnicGroup = st.slider('ethnicity', 0, 5)
ParentEduc = st.slider('parent education', 1, 6)
LunchType = st.slider('lunch', 0, 1)
TestPrep = st.slider('prep course', 1, 2)
ParentMaritalStatus = st.slider('parent marital status', 1, 2)
PracticeSport = st. slider('sports', 1, 3)
IsFirstChild = st.slider('first child', 0, 1)
NrSiblings = st. slider('siblings', 0, 6)
TransportMeans = st.slider('transportation', 1, 2)
WklyStudyHours = st.slider('weekly study hours', 1, 3)

input = np.array([Gender, EthnicGroup,	ParentEduc,	LunchType,	TestPrep,	ParentMaritalStatus	,PracticeSport,	IsFirstChild,	NrSiblings,	TransportMeans,	WklyStudyHours]).reshape(1, -1)
pred = loaded_model.predict(input)[0]



#Gender	EthnicGroup	ParentEduc	LunchType	TestPrep	ParentMaritalStatus	PracticeSport	IsFirstChild	NrSiblings	TransportMeans	WklyStudyHours	MathScore

if st.button('TEST!'):
    if pred == 0:
        st.header('This student will most likely need additional support and tutoring services.')
    if pred == 1:
        st.header('This student will most likely attain a grade above 70 on the state exam.')

st.divider()
    


