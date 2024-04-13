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
    return jsonify({'prediction': prediction.tolist()})  # Convert prediction array to list if necessary



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

@api.route('/retrain', methods=['POST'])
def retrain():
    result = retrain_model()
    return jsonify({'result': result})