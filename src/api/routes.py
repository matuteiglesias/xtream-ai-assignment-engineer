# api/routes.py
# import sys
# sys.path.append('/home/matias/repos/xtream-ai-assignment-engineer/src')
import os
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime

# from model.model import predict_price, retrain_model, get_models

import pandas as pd
import joblib


import mlflow
import glob


from data.data_preprocessing import *
from model.model import *
# from ..model.model import *


api = Blueprint('api', __name__)



@api.route('/predict', methods=['POST'])
def predict():
    print("Starting prediction endpoint...")

    # Assuming JSON data is posted
    data = request.get_json()
    print(f"Received data: {data}")

    model_name = data.get('model')  # Get the model name from the request
    print(f"Model name received: {model_name}")

    # Directory where the models and preprocessors are stored
    models_directory = os.path.join('model', 'models')
    temp_directory = current_app.config['TEMP_DIR']

    print(f"Model directory: {models_directory}")
    print(f"Temporary directory: {temp_directory}")

    # Attempt to find the latest preprocessor file
    preprocessor_files = glob.glob(os.path.join(temp_directory, 'preprocessor_*.joblib'))
    if not preprocessor_files:  # If no temp files, use default
        preprocessor_files = glob.glob(os.path.join(models_directory, 'preprocessor_default.joblib'))
    print(f"Found preprocessor files: {preprocessor_files}")
    
    latest_preprocessor_file = max(preprocessor_files, key=os.path.getctime, default=None)
    print(f"Using preprocessor file: {latest_preprocessor_file}")

    # Attempt to find the latest model file
    model_files = glob.glob(os.path.join(temp_directory, f'trained_model_*.joblib'))
    if not model_files:  # If no temp files, use default
        model_files = glob.glob(os.path.join(models_directory, f'trained_model_default.joblib'))
    print(f"Found model files: {model_files}")
    
    latest_model_file = max(model_files, key=os.path.getctime, default=None)
    print(f"Using model file: {latest_model_file}")

    if not latest_preprocessor_file or not latest_model_file:
        return jsonify({'error': 'Model or preprocessor not found'}), 404

    try:
        preprocessor = joblib.load(latest_preprocessor_file)
        model = joblib.load(latest_model_file)
        print("Model and preprocessor loaded successfully.")
    except Exception as e:
        return jsonify({'error': f'Error loading model or preprocessor: {str(e)}'}), 500

    # Prepare the data using the preprocessor
    X_preprocessed = preprocess_single_observation(data, latest_preprocessor_file)
    print(f"Data after preprocessing: {X_preprocessed}")

    # Make predictions using the loaded model
    prediction = model.predict(X_preprocessed)
    print(f"Prediction result: {prediction}")

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
    # List all joblib files in the default model directory that are not preprocessors
    default_models = [f for f in os.listdir(default_models_dir) if f.endswith('.joblib') and 'preprocessor' not in f]

    # Temporary models directory, should be stored in app config or as a global
    temp_dir = current_app.config['TEMP_DIR']
    # List all joblib files in the temporary directory that are not preprocessors
    temp_models = [f for f in os.listdir(temp_dir) if f.endswith('.joblib') and 'preprocessor' not in f]

    # Combine both lists
    all_models = default_models + temp_models
    return jsonify({'models': all_models})



import cProfile
import pstats
import io
from line_profiler import LineProfiler

def profile_retrain():
    lp = LineProfiler()
    lp_wrapper = lp(retrain)
    lp_wrapper()
    lp.print_stats()

try:
    profile  # Check if profile is already defined (e.g., by kernprof)
except NameError:
    def profile(func):
        return func  # Return the function unchanged if not profiling


# conda env setting for MLFlow
conda_env = {
    'name': 'mlflow-env',
    'channels': ['defaults', 'conda-forge'],
    'dependencies': [
        'python=3.11.3',  # Match the Python version used
        'scikit-learn=1.3.0',  # Match the scikit-learn version used
        'numpy',  # Assuming numpy is a dependency; specify the version if needed
        'pandas',  # Assuming pandas is a dependency; specify the version if needed
        {'pip': [
            'mlflow==2.11.3',  # Match the MLflow version used
            'cloudpickle',  # Ensure cloudpickle is included if used for serialization
            # Include any other pip packages that your model or preprocessor specifically needs
        ]}
    ]
}



@api.route('/retrain', methods=['POST'])
@profile  # Add this decorator to the retrain function
def retrain():
    profiler = cProfile.Profile()
    profiler.enable()


    print("Current working directory:", os.getcwd())


    details = request.get_json()
    label = details.get('label', 'New')
    proportion_factor = details.get('proportionFactor', 2)
    n_samples = details.get('nSamples', 1500)  # Default to 150 if not provided


    # Generate and set the experiment name in MLflow
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # same timestamp will tag mlflow, model and preprocessor files.
    experiment_name = f"retrain_{timestamp}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_param("label", label)
        mlflow.log_param("proportion_factor", proportion_factor)





        # Load existing data and simulate new observations
        existing_data = pd.read_csv('./data/diamonds/diamonds.csv'); existing_data['Label'] = 'Standard'
        new_data = simulate_new_observations(existing_data, n_samples, label, proportion_factor)
        combined_data = pd.concat([existing_data, new_data]).reset_index(drop=True)

        # Preprocess the combined dataset
        X_preprocessed, y_preprocessed = preprocess_pipeline(combined_data, timestamp)


        model, model_metrics, y_test, y_pred = train_and_evaluate_model(X_preprocessed, y_preprocessed)

        # Log metrics (metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
        for metric_name, metric_value in model_metrics.items():
            mlflow.log_metric(metric_name, metric_value)


        # model_path = f'./model/models/trained_model_{timestamp}.joblib'
        temp_dir = current_app.config['TEMP_DIR']
        model_path = os.path.join(temp_dir, f'trained_model_{timestamp}.joblib')
        
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        joblib.dump(model, model_path)
        print(f"Model trained and saved as {model_path}")


        # Log model and preprocessor as artifacts
        mlflow.sklearn.log_model(
            model, 
            "model", 
            registered_model_name="DiamondPricePredictor",
            conda_env=conda_env  # Use the manually specified conda environment
        )
        mlflow.log_artifact(model_path, "model")
    
        # Save predictions to a CSV
        predictions_path = os.path.join(temp_dir, f"predictions_{run_id}.csv")
        pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}).to_csv(predictions_path, index=False)
        
        # Log predictions as an artifact
        mlflow.log_artifact(predictions_path)


    # profiler.disable()
    # s = io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    # ps.print_stats(20)
    # print(s.getvalue())


    # Return statement should be here, outside the 'with' context
    return jsonify({
        "message": "Data added and model retrained",
        "modelPath": model_path,
        "modelName": model_path.split('/')[-1],
        "predictionsPath": predictions_path,
        "run_id": run_id
    })



import matplotlib.pyplot as plt
from io import BytesIO
import base64
import glob





@api.route('/plot-data', methods=['GET'])
def plot_predictions():
    run_id = request.args.get('run_id')
    print('Run ID:', run_id)  # Log the run ID being used

    if not run_id:
        print('Run ID is undefined. Returning error.')
        return jsonify({'error': 'Run ID is undefined'}), 400

    # Use glob to find the file
    predictions_files = glob.glob(f'./mlruns/**/predictions_{run_id}.csv', recursive=True)
    if not predictions_files:
        return jsonify({'error': 'Predictions file not found'}), 404
    
    predictions_path = predictions_files[0]  # Assuming the first match is what we want
    print('Predictions Path:', predictions_path)  # Log the predictions file path


    predictions_df = pd.read_csv(predictions_path)
    print('Predictions DataFrame:', predictions_df)  # Log the predictions DataFrame

    y_test, y_pred = predictions_df['y_test'], predictions_df['y_pred']

    # Generate plot
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.3, s = 3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.title('Predicted vs. Actual Prices')
    plt.grid(True)

    # Convert plot to image file
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return jsonify({'image': img_base64})






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



from flask import request, jsonify
import requests
import time
import json

@api.route('/test_performance', methods=['POST'])
def test_performance():
    data = request.get_json()  # Get JSON data from the POST request
    endpoint = data.get('endpoint', 'predict')  # Get the endpoint from JSON, default to 'predict'
    
    num_requests = 5
    start_time = time.time()

    if endpoint == 'predict':
        url_to_test = 'http://localhost:5000/api/predict'
        test_data = json.dumps({
            "carat": 1,
            "cut": "Ideal",
            "color": "G",
            "clarity": "SI1",
            "depth": 62,
            "table": 56,
            "x": 6,
            "y": 6,
            "z": 4,
            "model": "trained_model_default.joblib"
        })
    elif endpoint == 'retrain':
        url_to_test = 'http://localhost:5000/api/retrain'
        test_data = json.dumps({
            "label": "New",
            "proportionFactor": 2
        })
    else:
        return jsonify({'error': 'Invalid endpoint specified'}), 400

    headers = {'Content-type': 'application/json'}
    responses = []
    for _ in range(num_requests):
        response = requests.post(url_to_test, data=test_data, headers=headers)
        if response.status_code != 200:
            responses.append(response.status_code)

    end_time = time.time()
    duration = end_time - start_time
    response_time = duration / num_requests
    throughput = num_requests / duration

    return jsonify({
        'response_time': response_time,
        'throughput': throughput,
        'failed_responses': responses  # Adding this to track any failed responses during the test
    })

