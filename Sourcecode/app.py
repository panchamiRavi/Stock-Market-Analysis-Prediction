from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
import os
import warnings
from datetime import datetime, timedelta
import json

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connection

# ===== CONFIG =====
DATA_PATH = "C:/Users/user/Desktop/EconomyFinanceproject/nse_cleaned.csv"  # Update this path as needed
MODELS_DIR = "models"  # Directory to store trained models

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Global variable to store loaded data
df = None

def load_data():
    """Load and preprocess the NSE data"""
    global df
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.strip()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values(by=['symbol', 'date'])
        
        # Fill numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'deliverable_volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(method='ffill')
        
        # Feature engineering
        df['MA5'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=5).mean())
        df['MA10'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=10).mean())
        df = df.dropna()
        
        print("✅ Data loaded and preprocessed successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def train_model_for_symbol(symbol):
    """Train and save model for a specific symbol"""
    try:
        symbol = symbol.upper().strip()
        
        if symbol not in df['symbol'].unique():
            return None, "Symbol not found in dataset"
        
        stock_df = df[df['symbol'] == symbol].copy()
        
        # Feature engineering
        features = ['open', 'high', 'low', 'volume', 'deliverable_volume', 'MA5', 'MA10']
        target = 'close'
        
        X = stock_df[features]
        y = stock_df[target]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, shuffle=False
        )
        
        # Train model
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model and scaler
        model_path = os.path.join(MODELS_DIR, f"xgb_model_{symbol}.pkl")
        scaler_path = os.path.join(MODELS_DIR, f"scaler_{symbol}.pkl")
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred))
        }
        
        return metrics, "Model trained successfully"
        
    except Exception as e:
        return None, f"Error training model: {str(e)}"

def load_model_for_symbol(symbol):
    """Load trained model and scaler for a symbol"""
    try:
        symbol = symbol.upper().strip()
        model_path = os.path.join(MODELS_DIR, f"xgb_model_{symbol}.pkl")
        scaler_path = os.path.join(MODELS_DIR, f"scaler_{symbol}.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, None, "Model not found. Please train the model first."
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler, "Model loaded successfully"
        
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

@app.route('/')
def index():
    """Serve the welcome page"""
    try:
        # Read the welcome page content
        with open('welcome.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return """
        <h1>Stock Prediction API</h1>
        <p>Welcome page not found. Please ensure 'welcome.html' is in the same directory.</p>
        <h2>Available Endpoints:</h2>
        <ul>
            <li>GET / - Welcome page</li>
            <li>GET /dashboard - Prediction dashboard</li>
            <li>GET /api/symbols - Get available symbols</li>
            <li>POST /api/train - Train model for a symbol</li>
            <li>POST /api/predict - Make predictions</li>
        </ul>
        """

@app.route('/dashboard')
def dashboard():
    """Serve the prediction dashboard"""
    try:
        # Read the prediction dashboard content
        with open('prediction.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return """
        <h1>Prediction Dashboard Not Found</h1>
        <p>Dashboard HTML file not found. Please ensure 'prediction.html' is in the same directory.</p>
        <p><a href="/">← Back to Welcome Page</a></p>
        """

@app.route('/api/symbols', methods=['GET'])
def get_symbols():
    """Get list of available symbols"""
    try:
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        symbols = df['symbol'].unique().tolist()
        return jsonify({'symbols': symbols})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train model for a specific symbol"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
        
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        metrics, message = train_model_for_symbol(symbol)
        
        if metrics is None:
            return jsonify({'error': message}), 400
        
        return jsonify({
            'message': message,
            'symbol': symbol,
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions for a symbol"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
        
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        # Load model
        model, scaler, message = load_model_for_symbol(symbol)
        if model is None:
            # Try to train model automatically
            print(f"Model not found for {symbol}, training new model...")
            metrics, train_message = train_model_for_symbol(symbol)
            if metrics is None:
                return jsonify({'error': train_message}), 400
            
            # Load the newly trained model
            model, scaler, message = load_model_for_symbol(symbol)
            if model is None:
                return jsonify({'error': message}), 500
        
        # Get stock data
        stock_df = df[df['symbol'] == symbol].copy()
        stock_df = stock_df.sort_values('date')
        
        # Prepare features
        features = ['open', 'high', 'low', 'volume', 'deliverable_volume', 'MA5', 'MA10']
        
        # Get the last 30 days of data for visualization
        last_30_days = stock_df.tail(30).copy()
        
        # Prepare data for prediction
        X = last_30_days[features]
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Create response data
        dates = last_30_days['date'].dt.strftime('%Y-%m-%d').tolist()
        actual_values = last_30_days['close'].tolist()
        predicted_values = predictions.tolist()
        
        # Calculate metrics for the prediction period
        actual_array = np.array(actual_values)
        pred_array = np.array(predicted_values)
        
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(actual_array, pred_array))),
            'mae': float(mean_absolute_error(actual_array, pred_array)),
            'r2': float(r2_score(actual_array, pred_array))
        }
        
        # Predict next day
        latest_features = X_scaled[-1].reshape(1, -1)
        next_prediction = model.predict(latest_features)[0]
        
        return jsonify({
            'symbol': symbol,
            'labels': dates,
            'actual': actual_values,
            'predicted': predicted_values,
            'metrics': metrics,
            'next_prediction': float(next_prediction),
            'last_date': dates[-1] if dates else None,
            'message': 'Prediction completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Check API status"""
    return jsonify({
        'status': 'running',
        'data_loaded': df is not None,
        'available_symbols': len(df['symbol'].unique()) if df is not None else 0
    })

if __name__ == '__main__':
    print("🚀 Starting Stock Prediction API...")
    
    # Load data on startup
    if load_data():
        print(f"📊 Loaded data for {len(df['symbol'].unique())} symbols")
        print("🌐 Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load data. Please check the DATA_PATH configuration.")