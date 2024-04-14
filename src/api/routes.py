# api/routes.py

import sys
sys.path.append('/home/matias/repos/xtream-ai-assignment-engineer/src')
import os
from flask import Blueprint, request, jsonify
from model.model import predict_price, retrain_model, get_models

api = Blueprint('api', __name__)

# @api.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json(force=True)
#     prediction = predict_price(data)  # 'data' can be passed directly
#     return jsonify({'prediction': prediction.tolist()})  # Assuming prediction is a numpy array

# @api.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     prediction = predict_price(data)
#     return jsonify({'price': prediction})

import pandas as pd
import numpy as np
import joblib

from flask import Blueprint, request, jsonify
import pandas as pd
import os
import joblib

api = Blueprint('api', __name__)

@api.route('/predict', methods=['POST'])
def predict():
    # Assuming JSON data is posted
    data = request.get_json()
    model_name = data.get('model')  # Get the model name from the request
    model_path = os.path.join('model/models', model_name)
    preprocessor_path = os.path.join('model/models', 'preprocessor.joblib')

    # Load preprocessor and model
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)

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



# @api.route('/predict', methods=['POST'])
# def predict():
#     # your prediction logic here
#     return jsonify({'message': 'Prediction successful'})

# @api.route('/api/predict', methods=['POST'])
# def predict():
#     # your prediction logic here
#     return jsonify({'message': 'api Prediction successful'})


@api.route('/models', methods=['GET'])
def models():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../model/models')
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    return jsonify({'models': model_files})


# @api.route('/model', methods=['GET'])
# def models():
#     model_list = get_models()
#     return jsonify({'models': model_list})

# @api.route('/retrain', methods=['POST'])
# def retrain():
#     result = retrain_model()
#     return jsonify({'result': result})




# @api.route('/api/plot-data', methods=['GET'])
# def plot_data():
#     model_name = request.args.get('model', 'trained_sgd_model')  # Default model name
#     predictions_file = f'./data/diamonds/predictions_{model_name}.csv'
    
#     try:
#         if not os.path.exists(predictions_file):
#             raise FileNotFoundError(f"No predictions file found for model '{model_name}'.")
        
#         # Load the saved predictions
#         predictions_data = pd.read_csv(predictions_file)
#         y_test = predictions_data['Actual Prices'].tolist()
#         y_pred = predictions_data['Predicted Prices'].tolist()
        
#         return jsonify({'actual': y_test, 'predicted': y_pred})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
