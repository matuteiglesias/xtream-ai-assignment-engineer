import pandas as pd
from data.data_preprocessing import preprocess_data
from model import train_and_save_model, predict_price
# Adjust the import statement 'from your_model_module' to match the actual name of your Python file containing the model functions

def test_model_flow():
    print("Starting data preprocessing...")
    # Note: Ensure your preprocess_data function is adjusted to return both X and y
    X, y = preprocess_data()  # X is preprocessed data, y is labels
    
    print("Data preprocessing completed.")
    
    print("Training and saving the model...")
    # Train the model and save it
    # rmse, r2 = train_and_save_model()  # Ensure train_and_save_model is implemented correctly to return metrics
    # print(f"Model trained. RMSE: {rmse}, R^2: {r2}")
    
    # print("Loading the trained model...")
    # model = load_model()  # Make sure the path is correct in the load_model function
    # print("Model loaded successfully.")
    
    # Assuming you have the original dataset to test predictions (or you can split your dataset beforehand and save the test set)
    original_data = pd.read_csv('datasets/diamonds/diamonds.csv')

    # When making a prediction, ensure the data is in the correct form
    print("Making predictions on the first ten samples of the dataset...")
    for i in range(10):
        # Extract the raw features for prediction
        features_raw = original_data.iloc[[i]].drop('price', axis=1)
        prediction = predict_price(features_raw)
        print(f"Sample {i+1}, Prediction: {prediction[0]}")
        
    print("Test flow completed.")

if __name__ == "__main__":
    test_model_flow()
