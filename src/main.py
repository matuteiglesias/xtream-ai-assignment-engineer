from flask import Flask, jsonify, request, render_template
import os
import tempfile
import atexit
import mlflow
from flask_swagger_ui import get_swaggerui_blueprint

from api.routes import api  # Importing the Blueprint named 'api'

# Initialize Flask application
app = Flask(__name__, static_folder='static')


base_path = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where the script is located
mlruns_path = os.path.join(base_path, "mlruns")

if not os.path.exists(mlruns_path):
    os.makedirs(mlruns_path)
    print(f"Created mlruns directory at {mlruns_path}")

mlflow.set_tracking_uri(f"file://{mlruns_path}")
print("MLflow Tracking URI set to:", mlflow.get_tracking_uri())


# Swagger UI configuration
SWAGGER_URL = '/swagger'  # URL for exposing Swagger UI (without trailing '/')
API_URL = '/static/swagger/swagger.yaml'  # URL for Swagger spec
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
temp_dir = tempfile.mkdtemp(prefix="ml_")
app.config['TEMP_DIR'] = temp_dir


# Cleanup: Function to remove the temporary directory when the app is terminated
import shutil

def cleanup_temp_dir():
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Temporary directory cleaned up.")

atexit.register(cleanup_temp_dir)


# Cleanup at every request
# @app.teardown_appcontext
# def teardown_app(exception=None):
#     cleanup_temp_dir()


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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({"error": "Internal server error"}), 500

@app.route('/logs', methods=['GET'])
def view_logs():
    # Ensure you handle the security implications of exposing logs.
    log_content = open('application.log', 'r').read()
    return jsonify({"log": log_content}), 200



if __name__ == '__main__':
    app.run(debug=True)

