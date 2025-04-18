import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred):
    """Calculate and return evaluation metrics for regression"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2)
    }

def train_linear_regression(X_train, y_train, X_test, y_test):
    """Train a Linear Regression model and evaluate it"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate model
    metrics = evaluate_model(y_test, predictions)
    
    return model, predictions, metrics

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train a Random Forest model and evaluate it"""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate model
    metrics = evaluate_model(y_test, predictions)
    
    return model, predictions, metrics

def save_models(lr_model, rf_model, scaler, feature_names, target_name, model_dir='static/models'):
    """Save trained models and metadata to disk"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Save models
    joblib.dump(lr_model, os.path.join(model_dir, 'linear_regression_model.pkl'))
    joblib.dump(rf_model, os.path.join(model_dir, 'random_forest_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    # Save feature names and target
    metadata = {
        'feature_names': feature_names,
        'target_name': target_name
    }
    joblib.dump(metadata, os.path.join(model_dir, 'metadata.pkl'))
    
    return True

def load_models(model_dir='static/models'):
    """Load trained models and metadata from disk"""
    lr_model = joblib.load(os.path.join(model_dir, 'linear_regression_model.pkl'))
    rf_model = joblib.load(os.path.join(model_dir, 'random_forest_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    metadata = joblib.load(os.path.join(model_dir, 'metadata.pkl'))
    
    return lr_model, rf_model, scaler, metadata

def predict_price(input_data, model_name, feature_encoder, scaler):
    """
    Make a house price prediction based on user input
    
    Parameters:
    - input_data: dictionary of feature values
    - model_name: 'linear_regression' or 'random_forest'
    - feature_encoder: function to encode categorical features
    - scaler: fitted scaler to normalize features
    
    Returns:
    - predicted price
    """
    # Load models
    try:
        lr_model, rf_model, _, metadata = load_models()
        
        # Choose the appropriate model
        if model_name == 'linear_regression':
            model = lr_model
        else:  # random_forest
            model = rf_model
        
        # Transform input data into the correct format
        input_df = feature_encoder(input_data)
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        return float(prediction)
    
    except Exception as e:
        print(f"Error predicting price: {e}")
        return None

def get_algorithm_descriptions():
    """Return descriptions of the ML algorithms used in the project"""
    return {
        'linear_regression': {
            'name': 'Linear Regression',
            'description': 'Linear Regression is a statistical method that models the relationship between a dependent variable and one or more independent variables by fitting a linear equation. It assumes a linear relationship between variables.',
            'strengths': [
                'Simple and easy to interpret',
                'Fast training and prediction speed',
                'Works well with linearly separable data',
                'Requires minimal computational resources'
            ],
            'weaknesses': [
                'Cannot capture complex non-linear patterns',
                'Sensitive to outliers',
                'Assumes independence between features',
                'May underfit complex housing data'
            ]
        },
        'random_forest': {
            'name': 'Random Forest',
            'description': 'Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the average prediction of the individual trees. It captures non-linear patterns in the data.',
            'strengths': [
                'Handles non-linear relationships well',
                'Robust against overfitting',
                'Provides feature importance ranking',
                'Works well with both numerical and categorical features'
            ],
            'weaknesses': [
                'More computationally intensive than Linear Regression',
                'Less interpretable (black box model)',
                'May require more training data for optimal performance',
                'Prediction time can be slower than simpler models'
            ]
        }
    }

def get_feature_descriptions():
    """Return descriptions of the housing features"""
    return {
        'price': 'The sale price of the property in USD',
        'sqft': 'Square footage of the home',
        'bedrooms': 'Number of bedrooms',
        'bathrooms': 'Number of bathrooms (0.5 indicates a half-bath with toilet and sink but no shower/tub)',
        'neighborhood': 'Geographic area within the city where the property is located',
        'year_built': 'Year when the property was constructed',
        'lot_size': 'Size of the land in acres',
        'floors': 'Number of floors or levels in the home',
        'basement': 'Whether the home has a basement (1) or not (0)',
        'garage': 'Number of garage spaces',
        'distance_to_city_center': 'Distance to downtown/city center in miles',
        'school_rating': 'Average rating of nearby schools on a scale of 1-10',
        'property_type': 'Type of property (e.g., Single Family, Townhouse, Condo)'
    }