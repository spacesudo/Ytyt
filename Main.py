import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import timedelta

# Function to fetch OHLC data with headers
def fetch_ohlc_data(days: int = 90):
    """
    Fetch OHLC data for Bitcoin from CoinGecko API with headers.
    Args:
        days (int): Number of past days to fetch data.
    Returns:
        pd.DataFrame: OHLC data.
    """
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": str(days),
        "interval": "daily"
    }
    headers = {
        "User-Agent": "CryptoPredictor/1.0",  # Replace with your app details if necessary
        # Add any other necessary headers here (e.g., API keys if required)
    }
    response = requests.get(url, params=params, headers=headers)
    
    # Check for errors
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")

    data = response.json()
    
    # Parse OHLC data
    prices = data.get('prices', [])
    df = pd.DataFrame(prices, columns=["timestamp", "close"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Multi-day prediction function
def predict_future_prices(df: pd.DataFrame, days_to_predict: int):
    """
    Predict future prices for a given number of days.
    Args:
        df (pd.DataFrame): DataFrame with historical prices.
        days_to_predict (int): Number of days to predict.
    Returns:
        pd.DataFrame: Predicted dates and prices for the specified number of days.
    """
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
ohlc_data = fetch_ohlc_data(days=90)
predicted_prices = predict_future_prices(ohlc_data, days_to_predict=5)

print("Predicted Prices for the Next 5 Days:")
print(predicted_prices)
