import requests
import json

# The URL where your Flask API is running
API_URL = "http://localhost:5000/api/predict"

# Simulated JSON input
simulated_json = """
{
    "carat": 1.1,
    "cut": "Ideal",
    "color": "H",
    "clarity": "SI2",
    "depth": 62,
    "table": 55,
    "x": 6.61,
    "y": 6.65,
    "z": 4.11
}
"""

# Convert the string to a dictionary
data = json.loads(simulated_json)

# Send a POST request to the API
response = requests.post(API_URL, json=data)

# Check if the request was successful
if response.status_code == 200:
    # Print the prediction from the response
    print("Prediction received from the API:")
    print(response.json())
else:
    print(f"Failed to get a prediction. Status code: {response.status_code}")
    print("Response message:", response.text)

# The output should be similar to:
# Prediction received from the API:
# {'prediction': [6.0]}
    