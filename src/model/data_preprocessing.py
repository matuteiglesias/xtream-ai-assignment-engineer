import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib


def preprocess_data():
    # Load data
    data = pd.read_csv('datasets/diamonds/diamonds.csv')
    
    # Separate target variable 'y' (price) and features 'X'
    y = data['price']  # Target variable
    X = data.drop(columns=['price'])  # Features matrix
    
    # Preprocessing steps
    numeric_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
    categorical_features = ['cut', 'color', 'clarity']
    
    # Numeric features adjustments
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('power', PowerTransformer(method='yeo-johnson'))
    ])
    
    # Categorical features adjustments
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder())
    ])
    
    # Combine transformations
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # Apply preprocessing
    X_preprocessed = preprocessor.fit_transform(X)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    
    # Optionally, convert the output back to a DataFrame
    columns_transformed = (numeric_features + 
                           list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
    X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=columns_transformed)
    
    return X_preprocessed, y
