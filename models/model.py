from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
# from data_preprocessing import preprocess_data  # Adjust import path as needed
from .data_preprocessing import preprocess_data
import joblib
import numpy as np
import pandas as pd

def train_and_save_model():
    # Assuming preprocess_data() function returns a preprocessed features matrix X and labels vector y
    X, y = preprocess_data()
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the RandomForestRegressor
    model = RandomForestRegressor(random_state=42)
    
    # Define a grid of hyperparameters to search over
    param_grid = {
        'n_estimators': [100, 200],  # Number of trees in the forest
        'max_depth': [None, 10, 20],  # Maximum depth of the tree
        'min_samples_split': [2, 5],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2]  # Minimum number of samples required to be at a leaf node
    }
    
    # Setup the grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    
    # Retrieve the best model from grid search
    best_model = grid_search.best_estimator_
    
    # Predict on the testing set
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R^2 Score: {r2}")
    
    # Save the best model
    joblib.dump(best_model, 'models/trained_model.joblib')

    # Optionally return the metrics for external use
    return rmse, r2

# Assuming 'preprocessor' is your ColumnTransformer instance from training
# You need to save and load this preprocessor along with your model


# Assuming the preprocessor and model are saved in the 'models/' directory.
PREPROCESSOR_PATH = 'models/preprocessor.joblib'
MODEL_PATH = 'models/trained_model.joblib'

def load_resources():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model

def predict_price(features_raw):
    preprocessor, model = load_resources()
    
    # Transform features_raw into a DataFrame if it's not already
    if not isinstance(features_raw, pd.DataFrame):
        features_raw = pd.DataFrame(features_raw, index=[0])
    
    features_preprocessed = preprocessor.transform(features_raw)
    prediction = model.predict(features_preprocessed)
    return prediction