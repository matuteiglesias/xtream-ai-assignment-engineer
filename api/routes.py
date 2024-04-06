# api/routes.py
from flask import Blueprint, request, jsonify
from models.model import predict_price

api = Blueprint('api', __name__)

@api.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = predict_price(data)  # 'data' can be passed directly
    return jsonify({'prediction': prediction.tolist()})  # Assuming prediction is a numpy array
