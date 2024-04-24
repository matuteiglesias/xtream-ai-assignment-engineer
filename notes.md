
# Project Development Memo

## Objective

The goal of this project is to develop a robust and scalable machine learning API that predicts diamond prices based on their features. This endeavor seeks to blend comprehensive ML pipeline management, detailed preprocessing, and production-ready API.

## Main aspects covered:

- **Data Management:** Utilize a detailed data preprocessing approach aligning input data and model.
- **Model Training:** Explore advanced ML models, considering model performance and efficiency. Emphasis on evaluation metrics.
- **API Development:** Develop a Flask API with comprehensive endpoint management. The API will serve as the interface for model predictions.
- **Containerization:** Implement application containerization. Envision Cloud integration.


## How to Use the Diamond Price Prediction Application

### Accessing the Web Application

1. Navigate to [Web App URL] in your web browser. Available when the app is publicly deployed.
2. Fill in the required fields with the diamond's characteristics (e.g., carat, cut, color, clarity, etc.).
3. Click the "Predict" button to submit the information.
4. You will receive an estimated price for the diamond based on the input features.
5. You can add a batch of new data. Choose a label for the new batch. The prices in this batch will be larger than typical diamond prices by a proportion factor you input. The model is retrained and you can use it to predict prices from any batch.

### Running Locally

If you'd like to run the application locally, ensure you have Docker installed, git as well, then:

1. Clone the repository: Execute `git clone -b localdev https://github.com/matuteiglesias/xtream-ai-assignment-engineer.git` to download the project files to your local machine.
2. Navigate to the main directory: `cd xtream-ai-assignment-engineer/src`
3. Build the Docker image: `sudo docker build -t diamond-price-predictor .`
4. Run the container: `sudo docker run -p 5000:5000 diamond-price-predictor`
5. Access the application at `http://localhost:5000` in your browser.


**Additional Resources:**
- API Documentation: Access the Swagger UI for API documentation at `http://localhost:5000/swagger`.


 <!-- Detached mode -->
 <!-- sudo docker run -d -p 5000:5000 diamond-price-predictor -->
