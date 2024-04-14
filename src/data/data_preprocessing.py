
import sys
sys.path.append('/home/matias/repos/xtream-ai-assignment-engineer/src')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from datetime import datetime


def initialize_default_preprocessor():
    data = pd.read_csv('./data/diamonds/diamonds.csv')
    if 'Label' not in data.columns:
        data['Label'] = 'Standard'
    
    y = data['price']
    X = data.drop(columns=['price'])
    
    numeric_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
    categorical_features = ['cut', 'color', 'clarity', 'Label']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    preprocessor.fit(X)
    joblib.dump(preprocessor, './model/models/default_preprocessor.joblib')
    return preprocessor

# def update_preprocessor_with_new_data(new_data):
#     existing_data = pd.read_csv('./data/diamonds/diamonds.csv')
#     combined_data = pd.concat([existing_data, new_data])
    
#     preprocessor = joblib.load('./model/models/default_preprocessor.joblib')
#     preprocessor.fit(combined_data.drop('price', axis=1))
    
#     new_preprocessor_path = './model/models/updated_preprocessor_{}.joblib'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
#     joblib.dump(preprocessor, new_preprocessor_path)
#     return preprocessor


import os

def preprocess_data():
    data_path = './data/diamonds/diamonds.csv'
    preprocessor_path = './model/models/default_preprocessor.joblib'
    
    data = pd.read_csv(data_path)
    if 'Label' not in data.columns:
        data['Label'] = 'Standard'
    
    y = data['price']
    X = data.drop(columns=['price'])

    # Check if preprocessor exists, if not initialize and save default
    if not os.path.exists(preprocessor_path):
        preprocessor = initialize_default_preprocessor()
    else:
        preprocessor = joblib.load(preprocessor_path)
    
    X_preprocessed = preprocessor.transform(X)
    return X_preprocessed, y


    # # Preprocessing steps
    # numeric_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
    # categorical_features = ['cut', 'color', 'clarity', 'Label']


    # Optionally, convert the output back to a DataFrame
    # columns_transformed = (numeric_features + 
    #                        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
    # X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=columns_transformed)
    