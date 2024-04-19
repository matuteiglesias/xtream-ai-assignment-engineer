
import sys
sys.path.append('/home/matias/repos/xtream-ai-assignment-engineer/src')

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib 
import os
from flask import current_app



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# import joblib
# from datetime import datetime

def convert_categorical(data, categorical_cols):
    data[categorical_cols] = data[categorical_cols].astype('category')
    return data

def remove_outliers(data):
    data = data[(data['x'] != 0) & (data['y'] != 0) & (data['z'] != 0)]
    if 'price' in data.columns: data = data[(data['price'] >= 200) & (data['price'] <= 18010)]
    data['carat'] = data['carat'].astype(float)
    data = data[(data['carat'] > 0) & (data['carat'] <= 3.1)]
    return data

def add_features(data):
    # Make the x, y, z numeric
    data['x'] = data['x'].astype(float)
    data['y'] = data['y'].astype(float)
    data['z'] = data['z'].astype(float)

    data['rectangularness'] = np.abs(np.log10(data['x'] / data['y']))
    data = data[data['rectangularness'] < 0.015]
    data = data.drop(['x', 'y', 'z'], axis=1)
    return data

def prepare_features(data, categorical_features, numeric_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    preprocessor.fit(data)
    return preprocessor

# def save_preprocessor(preprocessor, filename):
#     joblib.dump(preprocessor, filename)

def preprocess_pipeline(data, timestamp):
   
    if 'Label' not in data.columns:
        data['Label'] = 'Standard'

    data = convert_categorical(data, ['cut', 'color', 'clarity', 'Label'])
    data = remove_outliers(data)
    data = add_features(data)


    X = data.drop('price', axis=1)
    y = np.log10(data['price'])
    
    numeric_features = X.columns.difference(['cut', 'color', 'clarity', 'Label']).tolist()
    preprocessor = prepare_features(X, ['cut', 'color', 'clarity', 'Label'], numeric_features)
    X_preprocessed = preprocessor.transform(X)
    

    # model_path = f'./model/models/trained_model_{timestamp}.joblib'
    temp_dir = current_app.config['TEMP_DIR']
    processor_file_name = os.path.join(temp_dir, f'preprocessor_{timestamp}.joblib')
    
    joblib.dump(preprocessor, processor_file_name)
    print(f"Preprocessor saved as {processor_file_name}")
    
    return X_preprocessed, y

# # Use the preprocessing pipeline
# X_preprocessed, y_preprocessed = preprocess_pipeline('./data/diamonds/diamonds.csv')
# print("Data preprocessed and ready for model training.")



def preprocess_single_observation(data, preprocessor_path):
    """
    Processes a single observation using an existing preprocessor.
    
    Args:
    data (dict or DataFrame): The input data which can be a dictionary (from JSON) or a DataFrame.
    preprocessor_path (str): The path to the saved preprocessor joblib file.
    
    Returns:
    DataFrame: The preprocessed features ready for prediction.
    """
    # Convert data to DataFrame if it's a dictionary
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    # Apply outlier and feature engineering
    data = remove_outliers(data)
    data = add_features(data)

    # Ensure 'Label' column is present
    if 'Label' not in data.columns:
        data['Label'] = 'Standard'

    # Handle categorical data conversion
    categorical_cols = ['cut', 'color', 'clarity', 'Label']
    data[categorical_cols] = data[categorical_cols].astype('category')


    # Drop 'price' column if present, as it's not used for predictions
    if 'price' in data.columns:
        data.drop('price', axis=1, inplace=True)
    
    # Load the preprocessor from the specified path
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)


    # # Print categories from OneHotEncoder within the preprocessor
    # cat_encoder = preprocessor.named_transformers_['cat'].categories_
    # print("Categories recognized by the encoder:")
    # for categories in cat_encoder.categories_:
    #     print(categories)

    # Extracting OneHotEncoder from the ColumnTransformer
    cat_encoder = preprocessor.named_transformers_['cat']
    print("Categories recognized by the encoder:")
    for categories in cat_encoder.categories_:
        print(categories)

    # Preprocess the features
    X_preprocessed = preprocessor.transform(data)
    
    return X_preprocessed




def simulate_new_observations(data, size, label, proportion_factor, save=False):
    """
    Simulates new observations by sampling from the existing dataset and adjusting the prices based on labels.

    Parameters:
        data (DataFrame): The original dataset.
        size (int): Number of samples to generate.
        label (str): Label to assign to the new samples ('special', 'bad quality', etc.).
        proportion_factor (float): Factor to adjust the prices (2 for doubling, 0.5 for halving).
        save (bool): Option to save the modified dataset to a CSV file.

    Returns:
        DataFrame: A new dataset with adjusted prices and labels.
    """

    # Randomly sample 'size' observations from the original dataset
    new_data = data.sample(n=size, random_state=42).reset_index(drop=True)

    # Add a new column 'Label' and assign the given label
    new_data['Label'] = label

    # Adjust the prices based on the label and proportion_factor
    new_data['price'] = new_data['price'] * proportion_factor

    # Save the new dataset if requested
    if save:
        file_name = f'datasets/modified_{label}_{int(proportion_factor*100)}.csv'
        new_data.to_csv(file_name, index=False)
        print(f"Dataset saved as {file_name}")

    return new_data
