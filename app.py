from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)
CORS(app)

model = None 
try:
    model = tf.keras.models.load_model('latest_model.h5')
    scaler = joblib.load('scaler.save')
    print("LSTM model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Define a function to load the dataset
TODAY = date.today()
START = (TODAY - timedelta(days=365)).strftime("%Y-%m-%d")
TODAY = date.today().strftime("%Y-%m-%d")

# Define a function to load the dataset
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

def process_raw(data): 
    data = data.iloc[:, 4:5].values
    data = scaler.fit_transform(data)
    input = []
    input.append(data[-100:])
    input = np.array(input) 
    return input

def forecast(input_data, model): 
    scaled_pred = model.predict(input_data)
    unscaled_pred = scaler.inverse_transform([[scaled_pred[0][0]]])[0][0]
    return unscaled_pred


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON payload from client
        data = request.get_json()
        ticker = data.get("ticker", None)

        if not ticker:
            return jsonify({"error": "Ticker symbol is required"}), 400

        # Load and preprocess data
        raw_data = load_data(ticker)
        if raw_data.empty:
            return jsonify({"error": f"No data found for ticker '{ticker}'"}), 404

        latest_price = float(raw_data["Close"].iloc[-1])
        latest_date = str(raw_data["Date"].iloc[-1].date())  
        print(latest_price)

        input_data = process_raw(raw_data)
        prediction = forecast(input_data, model)

        return jsonify({
            "ticker": ticker,
            "forecast": round(prediction, 2),
            "latest_price": round(latest_price, 2),
            "latest_date": latest_date
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)