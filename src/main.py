from flask import Flask, jsonify, request, render_template
import os
import tempfile
import atexit
import pandas as pd
import joblib
from datetime import datetime
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from flask_swagger_ui import get_swaggerui_blueprint

from api.routes import api  # Importing the Blueprint named 'api'

# Initialize Flask application
app = Flask(__name__, static_folder='static')

# Swagger UI configuration
SWAGGER_URL = '/swagger'  # URL for exposing Swagger UI (without trailing '/')
API_URL = '/static/swagger.yaml'  # URL for Swagger spec
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Diamond Price Prediction API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
app.register_blueprint(api, url_prefix='/api')  # Register the API Blueprint

# Temporary directory for session models
temp_dir = tempfile.mkdtemp()

# Cleanup: Function to remove the temporary directory when the app is terminated
def cleanup_temp_dir():
    os.rmdir(temp_dir)

atexit.register(cleanup_temp_dir)


@app.route('/api/retrain', methods=['POST'])
def retrain():
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
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
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

    model_path = f'./model/models/trained_model_{timestamp}.joblib'
    joblib.dump(model, model_path)
    print(f"Model trained and saved as {model_path}")
    # return jsonify({"message": "Data added and model retrained", "modelPath": model_path})
    return jsonify({"message": "Data added and model retrained", "modelPath": model_path, "modelName": model_path.split('/')[-1]})




def add_batch_features(data, batch_size, label, proportion_factor, save=False):
    new_data = data.sample(n=batch_size, random_state=42).reset_index(drop=True)
    new_data['Label'] = label
    new_data['price'] = new_data['price'] * proportion_factor
    data = pd.concat([data, new_data]).reset_index(drop=True)

    if save:
        file_name = f'./data/diamonds/modified_{label}_{int(proportion_factor*100)}.csv'
        data.to_csv(file_name, index=False)
        print(f"Dataset saved as {file_name}")

    return data



@app.route('/cleanup', methods=['POST'])
def cleanup():
    # Clean up the temporary directory
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        os.unlink(file_path)
    return jsonify({"message": "Temporary data cleaned up"})

@app.route('/')
def index():
    return render_template('index.html')

# Register the Blueprint with the app instance
app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True)




def save_model_predictions(model_name):
    # Directory paths
    data_directory = './data/diamonds'
    models_directory = './model/models'
    
    # Find the latest dataset
    latest_data_file = max([os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.csv')], key=os.path.getctime)
    
    # Load the latest data
    test_data = pd.read_csv(latest_data_file)
    y_test = test_data['price']
    X_test = test_data.drop('price', axis=1)
    
    # Load the specified model
    model_path = os.path.join(models_directory, f'model_{model_name}.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found with the name 'model_{model_name}.joblib'")
    model = joblib.load(model_path)
    
    # Predict prices
    y_pred = model.predict(X_test)
    
    # Save predictions along with actual prices and features
    results_df = pd.DataFrame(X_test)
    results_df['Actual Prices'] = y_test
    results_df['Predicted Prices'] = y_pred
    results_path = os.path.join(data_directory, f'predictions_{model_name}.csv')
    results_df.to_csv(results_path, index=False)

    return results_path  # Return the path for additional processing or logging



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
    
    # Load the specified model
    model_path = os.path.join(models_directory, f'{model_name}')
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
    print('X columns:', X_test.columns)

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


@app.route('/api/plot-data', methods=['GET'])
def plot_data():
    model_name = request.args.get('model')

    y_test, y_pred = get_latest_model_predictions(model_name)
    
    if not model_name:
        return jsonify({'error': 'Model name is undefined'}), 400
    
    try:
        y_test, y_pred = get_latest_model_predictions(model_name)
        return jsonify({'actual': y_test.tolist(), 'predicted': y_pred.tolist()})
    except Exception as e:
        # Log here if possible
        print(f"Error processing the prediction: {str(e)}")  # For debugging
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)

