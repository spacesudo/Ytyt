import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import timedelta

# Function to fetch OHLC data using the updated endpoint and headers
def fetch_ohlc_data(coin_id: str = "bitcoin", days: int = 1):
    """
    Fetch OHLC data for a cryptocurrency from CoinGecko API.
    Args:
        coin_id (str): The ID of the cryptocurrency (e.g., 'bitcoin').
        days (int): Number of days to fetch OHLC data for.
    Returns:
        pd.DataFrame: OHLC data with columns ['timestamp', 'open', 'high', 'low', 'close'].
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {
        "vs_currency": "usd",
        "days": str(days)
    }
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": "CG-PoMurepErqHySZn6VjNbxvND"  # Updated headers with API key
    }
    response = requests.get(url, params=params, headers=headers)
    
    # Check for errors
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")
    
    data = response.json()
    
    # Parse OHLC data
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convert timestamp
    return df

# Multi-day prediction function
def predict_future_prices(df: pd.DataFrame, days_to_predict: int):
    """
    Predict future prices for a given number of days.
    Args:
        df (pd.DataFrame): DataFrame with historical OHLC prices.
        days_to_predict (int): Number of days to predict.
    Returns:
        pd.DataFrame: Predicted dates and prices for the specified number of days.
    """
    # Use only 'close' price for prediction
    df['close_shifted'] = df['close'].shift(-1)
    df.dropna(inplace=True)

    X = df[['close']]
    y = df['close_shifted']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model (optional)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model RMSE: {rmse}")

    # Multi-day prediction
    predictions = []
    last_date = df['timestamp'].iloc[-1]  # Get the last available date
    current_input = [X.iloc[-1].values[0]]  # Start with the last close price

    for i in range(days_to_predict):
        next_prediction = model.predict([current_input])[0]
        next_date = last_date + timedelta(days=i + 1)  # Increment date
        predictions.append({"date": next_date, "predicted_price": next_prediction})
        current_input = [next_prediction]  # Use predicted value as input for next day

    return pd.DataFrame(predictions)

# Fetch data and make predictions
ohlc_data = fetch_ohlc_data(coin_id="bitcoin", days=30)  # Fetch 30 days of OHLC data
predicted_prices = predict_future_prices(ohlc_data, days_to_predict=5)  # Predict next 5 days

print("Predicted Prices for the Next 5 Days:")
print(predicted_prices)