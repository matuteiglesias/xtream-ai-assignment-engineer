
from data.data_preprocessing import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np


import os



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_and_evaluate_model(X, y, random_state=42, n_estimators=40, max_depth=9, min_samples_split=15, min_samples_leaf=2):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Initialize the model
    rf = RandomForestRegressor(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    
    # Train the model
    rf.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = rf.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Compile metrics
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    return rf, metrics, y_test, y_pred

# # Example of using the function with your preprocessed data
# model, model_metrics, y_true, y_predictions = train_and_evaluate_model(X_preprocessed, y_preprocessed)


# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'model', 'models', 'preprocessor.joblib')
# MODEL_PATH = os.path.join(BASE_DIR, 'model', 'models', 'trained_model.joblib')

# print(BASE_DIR, PREPROCESSOR_PATH, MODEL_PATH)

# def predict_price(features_raw, PREPROCESSOR_PATH, MODEL_PATH):
#     preprocessor = joblib.load(PREPROCESSOR_PATH)
#     model = joblib.load(MODEL_PATH)
    
#     # Transform features_raw into a DataFrame if it's not already
#     if not isinstance(features_raw, pd.DataFrame):
#         features_raw = pd.DataFrame(features_raw, index=[0])
    
#     features_preprocessed = preprocessor.transform(features_raw)
#     prediction = model.predict(features_preprocessed)
#     return prediction




import os

def get_models():
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    models = [file for file in os.listdir(model_dir) if file.endswith('.joblib')]
    return models
