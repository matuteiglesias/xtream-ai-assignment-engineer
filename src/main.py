from flask import Flask, render_template
from api.routes import api  # Make sure this is importing the Blueprint named 'api'

app = Flask(__name__, static_folder='static')


@app.route('/')
def index():
    return render_template('index.html')


# @app.route('/')
# def home():
#     return "Welcome to the Diamond Price Prediction API!"


# Register the Blueprint with the app instance
app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True)

