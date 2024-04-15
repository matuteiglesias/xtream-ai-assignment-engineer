# api/routes.py
import sys
sys.path.append('/home/matias/repos/xtream-ai-assignment-engineer/src')
import os
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime

# from model.model import predict_price, retrain_model, get_models

import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

import mlflow

api = Blueprint('api', __name__)

@api.route('/predict', methods=['POST'])
def predict():

    # Assuming JSON data is posted
    data = request.get_json()
    model_name = data.get('model')  # Get the model name from the request

    # Construct paths for the model
    permanent_model_path = os.path.join('model/models', model_name)
    temp_model_path = os.path.join(current_app.config['TEMP_DIR'], model_name)
    
    # Determine the correct model path
    if os.path.exists(permanent_model_path):
        model_path = permanent_model_path
    elif os.path.exists(temp_model_path):
        model_path = temp_model_path
    else:
        return jsonify({'error': 'Model not found'}), 404

    # Construct paths for the preprocessor
    preprocessor_name = 'preprocessor.joblib'  # This could also be dynamic if necessary
    permanent_preprocessor_path = os.path.join('model/models', preprocessor_name)
    temp_preprocessor_path = os.path.join(current_app.config['TEMP_DIR'], preprocessor_name)

    # Determine the correct preprocessor path
    if os.path.exists(permanent_preprocessor_path):
        preprocessor_path = permanent_preprocessor_path
    elif os.path.exists(temp_preprocessor_path):
        preprocessor_path = temp_preprocessor_path
    else:
        return jsonify({'error': 'Preprocessor not found'}), 404

    try:
        # Load preprocessor and model
        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
    except Exception as e:
        return jsonify({'error': f'Error loading model or preprocessor: {str(e)}'}), 500
    
    
    # Convert incoming JSON data to DataFrame
    df = pd.DataFrame([data])
    df.drop(columns=['model'], inplace=True)  # Remove the model entry from data

    # Ensure all expected columns are present
    expected_columns = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z', 'Label']
    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan  # Set missing columns to NaN or some default/derived value

    # Preprocess the data using the loaded preprocessor
    features_preprocessed = preprocessor.transform(df)

    # Make predictions using the loaded model
    prediction = model.predict(features_preprocessed)

    # Return the prediction in JSON format
    return jsonify({'price': prediction.tolist()})  # Convert prediction array to list if necessary


# @api.route('/models', methods=['GET'])
# def models():
#     model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../model/models')
#     model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
#     return jsonify({'models': model_files})


@api.route('/models', methods=['GET'])
def models():
    # Directory where default models are stored
    default_models_dir = './model/models'
    # List all joblib files in the default model directory
    default_models = [f for f in os.listdir(default_models_dir) if f.endswith('.joblib')]

    # Temporary models directory, should be stored in app config or as a global
    temp_dir = current_app.config['TEMP_DIR']
    # List all joblib files in the temporary directory
    temp_models = [f for f in os.listdir(temp_dir) if f.endswith('.joblib')]

    # Combine both lists
    all_models = default_models + temp_models
    return jsonify({'models': all_models})




@api.route('/retrain', methods=['POST'])
def retrain():


    print("Current working directory:", os.getcwd())


    details = request.get_json()
    label = details.get('label', 'New')
    proportion_factor = details.get('proportionFactor', 2)

    # Generate and set the experiment name in MLflow
    experiment_name = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_param("label", label)
        mlflow.log_param("proportion_factor", proportion_factor)



        print("Loading existing data...")
        # Load existing data
        data = pd.read_csv('./data/diamonds/diamonds.csv')
        if 'Label' not in data.columns: # Add a default label if not present
            data['Label'] = 'Standard'
        print(f"Original data shape: {data.shape}")
        # Get details from request
        details = request.get_json()
        label = details.get('label', 'New')
        proportion_factor = details.get('proportionFactor', 2)
        print(f"Received details from request - Label: {label}, Proportion Factor: {proportion_factor}")
        # Add new batch of data
        print("Adding new batch of data...")
        new_data = data.sample(n=100, random_state=42).reset_index(drop=True)
        new_data['Label'] = label
        new_data['price'] = new_data['price'] * proportion_factor
        data = pd.concat([data, new_data]).reset_index(drop=True)
        print(f"New data shape after adding batch: {data.shape}")
        # Assuming file path and check for preprocessor existence
        preprocessor_path = './model/models/default_preprocessor.joblib'
        if os.path.exists(preprocessor_path):
            print("Loading existing preprocessor...")
            preprocessor = joblib.load(preprocessor_path)
        else:
            # Create and fit a new preprocessor if not found
            print("Creating and fitting a new preprocessor...")
            categorical_features = ['cut', 'color', 'clarity', 'Label']
            numeric_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline([
                        ('scaler', StandardScaler())
                    ]), numeric_features),
                    ('cat', Pipeline([
                        ('encoder', OneHotEncoder(handle_unknown='ignore'))
                    ]), categorical_features)
                ])
            preprocessor.fit(data)
            print("Preprocessor fitted with new data.")

        # Accessing the correct pipeline and then the encoder
        print("Accessing the categorical pipeline and encoder...")
        cat_pipeline = preprocessor.named_transformers_['cat']
        encoder = cat_pipeline.named_steps['encoder']
        print("Fitting encoder with new categorical data...")
        encoder.fit(data[['cut', 'color', 'clarity', 'Label']])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save the updated preprocessor with timestamp in filename
        processor_file_name = f'./model/models/preprocessor_{timestamp}.joblib'
        temp_dir = current_app.config['TEMP_DIR']
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        joblib.dump(preprocessor, processor_file_name)
        print(f"Preprocessor saved as {processor_file_name}")

        # Retrain the model with the updated preprocessor and data
        print("Preparing data for model training...")
        X = data.drop('price', axis=1)
        y = data['price']
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        print("Transforming features using the updated preprocessor...")


        X_preprocessed = preprocessor.transform(X)
        print(f"Transformed features shape: {X_preprocessed.shape}")







        print("Retraining model...")
        model = SGDRegressor(random_state=42, 
                            loss='squared_error',
                            penalty='l1',
                            alpha=0.001,
                            l1_ratio=0.1,
                            learning_rate='adaptive',
                            max_iter=300,
                            tol=1e-3,
                            eta0=0.01)
        model.fit(X_preprocessed, y)

        # model_path = f'./model/models/trained_model_{timestamp}.joblib'
        temp_dir = current_app.config['TEMP_DIR']
        model_path = os.path.join(temp_dir, f'trained_model_{timestamp}.joblib')
        
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        joblib.dump(model, model_path)
        print(f"Model trained and saved as {model_path}")


        # Log model and preprocessor as artifacts
        mlflow.sklearn.log_model(model, "model", registered_model_name="DiamondPricePredictor")
        mlflow.log_artifact(model_path, "model")
        mlflow.log_artifact(processor_file_name, "preprocessor")
    
    return jsonify({"message": "Data added and model retrained", "modelPath": model_path, "modelName": model_path.split('/')[-1]})




def get_latest_model_predictions(model_name):
    # Directory paths
    data_directory = './data/diamonds'
    models_directory = './model/models'
    print('model name:', model_name)
    
    # Find the latest dataset
    latest_data_file = max([os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.csv')], key=os.path.getctime)
    
    # Load the latest data
    test_data = pd.read_csv(latest_data_file)
    y_test = test_data['price']
    X_test = test_data.drop('price', axis=1)
    
    # Adjust path to use TEMP_DIR
    temp_dir = current_app.config['TEMP_DIR']
    model_path = os.path.join(temp_dir, model_name)
    # model_path = os.path.join(models_directory, f'{model_name}')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    model = joblib.load(model_path)


    # Load the default preprocessor
    latest_processor_file = max([os.path.join(models_directory, f) for f in os.listdir(models_directory) if 'processor' in f], key=os.path.getctime)
    print('Latest processor file:', latest_processor_file)

    # preprocessor_path = os.path.join(models_directory, 'default_preprocessor.joblib')
    # preprocessor_path = os.path.join('./src/model/models', latest_processor_file)
    if not os.path.exists(latest_processor_file):
        raise FileNotFoundError(f"No preprocessor found at {latest_processor_file}")
    preprocessor = joblib.load(latest_processor_file)
    


    # Print details about the loaded preprocessor
    print("Loaded preprocessor from:", latest_processor_file)
    print(preprocessor)

    # For a ColumnTransformer
    if hasattr(preprocessor, 'transformers_'):
        print("ColumnTransformer details:")
        for transformer_name, transformer, columns in preprocessor.transformers_:
            print(f"Transformer: {transformer_name}")
            print(f" - Transformer object: {transformer}")
            print(f" - Columns: {columns}")
            if isinstance(transformer, Pipeline):
                for step_name, step in transformer.named_steps.items():
                    print(f"  - Step: {step_name}, Object: {step}")
                    if hasattr(step, 'categories_'):
                        print(f"    - Categories: {step.categories_}")


    # Preprocess the features
    print('X shape:', X_test.shape)
    print('X columns:', X_test.columns)  ######

    # X_test_preprocessed = preprocessor.transform(X_test)

    # Attempt to preprocess the features
    try:
        X_test_preprocessed = preprocessor.transform(X_test)
        print('X_test_preprocessed shape:', X_test_preprocessed.shape)
    except Exception as e:
        print("Error during preprocessing:", str(e))
        # Additional debugging to understand what went wrong
        if hasattr(preprocessor, 'transformers_'):
            for transformer_name, transformer, columns in preprocessor.transformers_:
                if isinstance(transformer, Pipeline):
                    # Check if the pipeline includes an encoder and if it's fitted
                    encoder = transformer.named_steps.get('encoder', None)
                    if encoder and hasattr(encoder, 'categories_'):
                        print(f"Encoder categories for {transformer_name}: {encoder.categories_}")
                    else:
                        print(f"No encoder or categories available in {transformer_name}")

    
    # Predict prices using the preprocessed features
    y_pred = model.predict(X_test_preprocessed)
    return y_test, y_pred


# temp_dir = current_app.config['TEMP_DIR']


@api.route('/plot-data', methods=['GET'])
def plot_data():
    model_name = request.args.get('model')
    if not model_name:
        return jsonify({'error': 'Model name is undefined'}), 400

    
    try:
        y_test, y_pred = get_latest_model_predictions(model_name)
        return jsonify({'actual': y_test.tolist(), 'predicted': y_pred.tolist()})
    except Exception as e:
        # Log here if possible
        print(f"Error processing the prediction: {str(e)}")  # For debugging
        return jsonify({'error': str(e)}), 500
    


from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


@api.route('/get-model-info', methods=['GET'])
def get_model_info():
    client = MlflowClient()
    model_details = []

    try:
        # Fetch all registered models and their details
        registered_models = client.search_registered_models()

        for rm in registered_models:
            for version in rm.latest_versions:
                # Attempt to fetch the run associated with each model version
                try:
                    run_data = client.get_run(version.run_id).data
                except MlflowException as e:
                    run_data = {"metrics": {}, "params": {}, "tags": {}}
                    print(f"Failed to fetch run data for run_id {version.run_id}: {str(e)}")

                # Compile information about each model version
                info = {
                    "name": rm.name,
                    "version": version.version,
                    "run_id": version.run_id,
                    "status": version.status,
                    "metrics": run_data.metrics,
                    "params": run_data.params,
                    "tags": run_data.tags
                }
                model_details.append(info)

    except MlflowException as e:
        return jsonify({"error": f"Failed to fetch model information: {str(e)}"}), 500

    return jsonify(model_details)



# def save_model_predictions(model_name):
#     # Directory paths
#     data_directory = './data/diamonds'
#     models_directory = './model/models'
    
#     # Find the latest dataset
#     latest_data_file = max([os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.csv')], key=os.path.getctime)
    
#     # Load the latest data
#     test_data = pd.read_csv(latest_data_file)
#     y_test = test_data['price']
#     X_test = test_data.drop('price', axis=1)
    
#     # Load the specified model
#     model_path = os.path.join(models_directory, f'model_{model_name}.joblib')
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"No model found with the name 'model_{model_name}.joblib'")
#     model = joblib.load(model_path)
    
#     # Predict prices
#     y_pred = model.predict(X_test)
    
#     # Save predictions along with actual prices and features
#     results_df = pd.DataFrame(X_test)
#     results_df['Actual Prices'] = y_test
#     results_df['Predicted Prices'] = y_pred
#     results_path = os.path.join(data_directory, f'predictions_{model_name}.csv')
#     results_df.to_csv(results_path, index=False)

#     return results_path  # Return the path for additional processing or logging



# def add_batch_features(data, batch_size, label, proportion_factor, save=False):
#     new_data = data.sample(n=batch_size, random_state=42).reset_index(drop=True)
#     new_data['Label'] = label
#     new_data['price'] = new_data['price'] * proportion_factor
#     data = pd.concat([data, new_data]).reset_index(drop=True)

#     if save:
#         file_name = f'./data/diamonds/modified_{label}_{int(proportion_factor*100)}.csv'
#         data.to_csv(file_name, index=False)
#         print(f"Dataset saved as {file_name}")

#     return data
