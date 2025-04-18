import os
import pandas as pd
import numpy as np
import json
import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import (
    train_linear_regression, 
    train_random_forest, 
    evaluate_model, 
    save_models, 
    load_models,
    predict_price,
    get_algorithm_descriptions,
    get_feature_descriptions
)

app = Flask(__name__)
app.secret_key = 'real_estate_prediction_app'  # For session management

# Configure upload settings
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure models directory exists
os.makedirs('static/models', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Check if models exist
    models_exist = os.path.exists('static/models/linear_regression_model.pkl') and \
                   os.path.exists('static/models/random_forest_model.pkl')
    
    return render_template('index.html', models_exist=models_exist)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read the dataset
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Store column names for feature selection
        columns = df.columns.tolist()
        
        # Store dataset statistics for display
        stats = {
            'rows': len(df),
            'columns': len(columns),
            'numerical_cols': len(df.select_dtypes(include=['int64', 'float64']).columns),
            'categorical_cols': len(df.select_dtypes(include=['object', 'category']).columns)
        }
        
        # Calculate basic statistics for numerical columns
        numerical_stats = {}
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            numerical_stats[col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median()
            }
        
        # Get value counts for categorical columns (top 5 values)
        categorical_stats = {}
        for col in df.select_dtypes(include=['object', 'category']).columns:
            categorical_stats[col] = df[col].value_counts().head(5).to_dict()
        
        return render_template('index.html', 
                              filename=filename, 
                              columns=columns,
                              data_preview=df.head().to_html(classes='table table-striped table-hover'),
                              stats=stats,
                              numerical_stats=numerical_stats,
                              categorical_stats=categorical_stats)
    
    return redirect(request.url)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    filename = request.form['filename']
    target_column = request.form['target_column']
    test_size = float(request.form['test_size'])
    features = request.form.getlist('features')
    
    # Load dataset
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if filename.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    
    # Prepare the data
    X = df[features]
    y = df[target_column]
    
    # Store original feature columns and their data types for later prediction
    session['feature_dtypes'] = {col: str(X[col].dtype) for col in X.columns}
    session['categorical_features'] = X.select_dtypes(include=['object', 'category']).columns.tolist()
    session['numerical_features'] = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # For categorical features, store unique values
    unique_values = {}
    for col in session['categorical_features']:
        unique_values[col] = df[col].unique().tolist()
    session['categorical_unique_values'] = unique_values
    
    # Handle categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression model
    lr_model, lr_predictions, lr_metrics = train_linear_regression(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Train Random Forest model
    rf_model, rf_predictions, rf_metrics = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Calculate additional metrics for model comparison
    lr_avg_error_percent = (np.abs(y_test.values - lr_predictions) / y_test.values).mean() * 100
    rf_avg_error_percent = (np.abs(y_test.values - rf_predictions) / y_test.values).mean() * 100
    
    # Save the trained models
    save_models(lr_model, rf_model, scaler, list(X.columns), target_column)
    
    # Save important data in session
    session['target_column'] = target_column
    session['features'] = features
    session['encoded_features'] = list(X.columns)
    
    # Get current date for display
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Prepare data for visualization
    actual_vs_predicted_lr = [{"actual": float(y_test.iloc[i]), "predicted": float(lr_predictions[i])} 
                            for i in range(min(50, len(y_test)))]  # Limit to 50 points for visualization
    
    actual_vs_predicted_rf = [{"actual": float(y_test.iloc[i]), "predicted": float(rf_predictions[i])} 
                            for i in range(min(50, len(y_test)))]
    
    # Calculate residuals for visualization
    lr_residuals = [{"actual": float(y_test.iloc[i]), "residual": float(y_test.iloc[i] - lr_predictions[i])}
                    for i in range(min(50, len(y_test)))]
    
    rf_residuals = [{"actual": float(y_test.iloc[i]), "residual": float(y_test.iloc[i] - rf_predictions[i])}
                   for i in range(min(50, len(y_test)))]
    
    # Get feature importance for Random Forest
    feature_importance = []
    if hasattr(rf_model, 'feature_importances_'):
        feature_importance = [{"feature": X.columns[i], 
                             "importance": float(rf_model.feature_importances_[i])} 
                             for i in range(len(X.columns))]
        # Sort by importance
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        # Limit to top 10 features
        feature_importance = feature_importance[:10]
    
    # Compare metrics
    metrics_comparison = {
        "Linear Regression": {**lr_metrics, "avg_error_percent": float(lr_avg_error_percent)},
        "Random Forest": {**rf_metrics, "avg_error_percent": float(rf_avg_error_percent)}
    }
    
    # Get algorithm descriptions
    algo_descriptions = get_algorithm_descriptions()
    
    # Get feature descriptions
    feature_descriptions = get_feature_descriptions()
    
    # Render results
    return render_template('result.html',
                          filename=filename,
                          target=target_column,
                          features=features,
                          lr_metrics=metrics_comparison["Linear Regression"],
                          rf_metrics=metrics_comparison["Random Forest"],
                          actual_vs_predicted_lr=json.dumps(actual_vs_predicted_lr),
                          actual_vs_predicted_rf=json.dumps(actual_vs_predicted_rf),
                          lr_residuals=json.dumps(lr_residuals),
                          rf_residuals=json.dumps(rf_residuals),
                          feature_importance=json.dumps(feature_importance),
                          metrics_comparison=json.dumps(metrics_comparison),
                          algo_descriptions=algo_descriptions,
                          feature_descriptions=feature_descriptions,
                          current_date=current_date)

@app.route('/home', methods=['GET'])
def home():
    """Home page with price prediction form after models are trained"""
    # Check if models exist
    try:
        lr_model, rf_model, _, metadata = load_models()
        
        # Get categorical features and their unique values
        categorical_features = session.get('categorical_features', [])
        categorical_unique_values = session.get('categorical_unique_values', {})
        numerical_features = session.get('numerical_features', [])
        
        return render_template('home.html', 
                              features=session.get('features', []),
                              target=session.get('target_column', 'price'),
                              categorical_features=categorical_features,
                              categorical_values=categorical_unique_values,
                              numerical_features=numerical_features,
                              feature_descriptions=get_feature_descriptions())
    except:
        return redirect(url_for('index'))

@app.route('/make_prediction', methods=['POST'])
def make_prediction():
    """Handle the prediction form submission"""
    try:
        # Get features from session
        features = session.get('features', [])
        
        # Get feature values from the form
        input_data = {}
        for feature in features:
            if feature in request.form:
                value = request.form[feature]
                
                # Convert to the right type
                dtype = session.get('feature_dtypes', {}).get(feature)
                if 'int' in dtype:
                    value = int(value)
                elif 'float' in dtype:
                    value = float(value)
                
                input_data[feature] = value
        
        # Get selected model
        model_name = request.form.get('model', 'random_forest')
        
        # Function to encode categorical features
        def encode_features(input_dict):
            # Create a DataFrame with one row
            input_df = pd.DataFrame([input_dict])
            
            # One-hot encode categorical features
            input_encoded = pd.get_dummies(input_df, drop_first=True)
            
            # Make sure all columns from training are present
            for col in session.get('encoded_features', []):
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Ensure columns order matches training
            return input_encoded[session.get('encoded_features', [])]
        
        # Load the scaler
        _, _, scaler, _ = load_models()
        
        # Make prediction
        predicted_price = predict_price(input_data, model_name, encode_features, scaler)
        
        # Format the price for display
        formatted_price = "${:,.2f}".format(predicted_price) if predicted_price else "Prediction Error"
        
        # Prepare input data for display
        display_input = {k: (v if k not in ['basement', 'garage'] else int(v)) for k, v in input_data.items()}
        
        # Get algorithm descriptions
        algo_descriptions = get_algorithm_descriptions()
        model_info = algo_descriptions.get(model_name, {})
        
        return render_template('prediction_result.html',
                              prediction=formatted_price,
                              raw_prediction=predicted_price,
                              input_data=display_input,
                              model_name=model_info.get('name', model_name.capitalize()),
                              model_info=model_info)
    
    except Exception as e:
        return render_template('error.html', error=f"Error making prediction: {str(e)}")

@app.route('/compare_models', methods=['GET'])
def compare_models():
    """Show detailed model comparison information"""
    try:
        # Get algorithm descriptions
        algo_descriptions = get_algorithm_descriptions()
        
        # Check if models are trained
        try:
            _, _, _, metadata = load_models()
            models_trained = True
        except:
            models_trained = False
        
        return render_template('model_comparison.html', 
                              algo_descriptions=algo_descriptions,
                              models_trained=models_trained)
    
    except Exception as e:
        return render_template('error.html', error=f"Error loading model comparison: {str(e)}")

@app.route('/download_models', methods=['GET'])
def download_models():
    """Return model information and download links"""
    return render_template('download_models.html')

@app.route('/about', methods=['GET'])
def about():
    """About page with project information"""
    return render_template('about.html')

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('error.html', error="Page not found. The requested URL does not exist."), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('error.html', error="Internal server error. Please try again later."), 500

if __name__ == '__main__':
    app.run(debug=True)