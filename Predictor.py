import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def str_to_datetime(s):
    """Convert string date to datetime object"""
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)

def download_msft_data(period="5y", interval="1d"):
    """Download MSFT stock data from Yahoo Finance"""
    print(f"Downloading {period} of MSFT stock data...")
    msft = yf.Ticker("MSFT")
    df = msft.history(period=period, interval=interval)
    
    # Reset index to make Date a column
    df = df.reset_index()
    
    # Save to CSV
    df.to_csv("MSFT.csv", index=False)
    
    print(f"Downloaded {len(df)} days of MSFT stock data")
    return df

def df_to_windowed_df(df, window_size=3):
    """Create a windowed dataframe for LSTM input"""
    print("Creating windowed dataframe...")
    
    # Convert to numpy for easier manipulation
    dates = df['Date'].values
    prices = df['Close'].values
    
    X, y = [], []
    windowed_dates = []
    
    # Create windows
    for i in range(len(prices) - window_size):
        X.append(prices[i:i+window_size])
        y.append(prices[i+window_size])
        windowed_dates.append(dates[i+window_size])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X for LSTM input [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return windowed_dates, X, y

def build_lstm_model(window_size):
    """Build the LSTM model architecture"""
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(window_size, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(X_train, y_train, X_val, y_val, window_size=3, epochs=100, batch_size=32):
    """Train the LSTM model"""
    print("Training LSTM model...")
    
    # Build model
    model = build_lstm_model(window_size)
    
    # Early stopping to prevent overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('msft_training_history.png')
    plt.close()
    
    print(f"Model trained for {len(history.history['loss'])} epochs")
    return model

def save_model_and_scaler(model, scaler):
    """Save the trained model and scaler"""
    # Save model in Keras format (compatible with Keras 3)
    model.save('msft_model.keras')
    
    # Save scaler
    with open('msft_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model and scaler saved successfully")

def predict_n_days_ahead(model, scaler, latest_data, n_days=10, window_size=3):
    """Predict stock prices for n days ahead"""
    predictions = []
    
    # Make a copy of the input data to avoid modifying the original
    current_window = np.copy(latest_data)
    scaled_window = scaler.transform(current_window.reshape(-1, 1)).reshape(window_size, 1)
    
    # Make predictions for n days
    for _ in range(n_days):
        # Reshape for LSTM input [samples, time steps, features]
        reshaped_window = scaled_window.reshape(1, window_size, 1)
        
        # Make prediction
        scaled_prediction = model.predict(reshaped_window, verbose=0)
        
        # Store the scaled prediction
        predictions.append(scaled_prediction[0][0])
        
        # Update the window for the next prediction (remove oldest, add new prediction)
        scaled_window = np.roll(scaled_window, -1, axis=0)
        scaled_window[-1] = scaled_prediction
    
    # Convert all predictions back to original scale
    original_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    
    return original_predictions

def get_future_dates(start_date, n_days=10):
    """Generate future business dates"""
    future_dates = []
    current_date = start_date
    
    while len(future_dates) < n_days:
        current_date += datetime.timedelta(days=1)
        # Skip weekends (5=Saturday, 6=Sunday)
        if current_date.weekday() < 5:
            future_dates.append(current_date)
    
    return future_dates

def plot_predictions(dates, prices, future_dates, predictions, window_size):
    """Plot historical prices and predictions"""
    plt.figure(figsize=(14, 7))
    
    # Get the last 90 days for plotting
    display_days = 90
    if len(dates) > display_days:
        display_dates = dates[-display_days:]
        display_prices = prices[-display_days:]
    else:
        display_dates = dates
        display_prices = prices
    
    # Plot historical data
    plt.plot(display_dates, display_prices, label='Historical Prices')
    
    # Highlight the window used for prediction
    window_start = len(dates) - window_size
    plt.plot(dates[window_start:], prices[window_start:], 'g-', linewidth=3, label=f'Window ({window_size} days)')
    
    # Plot predictions
    plt.plot(future_dates, predictions, 'r--', label='Forecasted Prices')
    plt.plot(future_dates, predictions, 'ro')
    
    # Highlight the latest actual price
    plt.plot(dates[-1], prices[-1], 'bo', markersize=8, label='Latest Close Price')
    
    plt.title('MSFT Stock Price Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price (USD)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    # Save the forecast chart
    plt.savefig('msft_forecast.png')
    plt.close()
    print("Forecast chart saved as 'msft_forecast.png'")

def print_forecast_table(latest_price, future_dates, forecast_prices):
    """Print the forecast in a table format"""
    print("\nMSFT Stock Price Forecast:")
    print("-" * 60)
    print(f"{'Date':<12} | {'Predicted Price':<15} | {'Change $':<10} | {'Change %':<10}")
    print("-" * 60)
    
    previous_price = latest_price
    
    for i, date in enumerate(future_dates):
        date_str = date.strftime('%Y-%m-%d')
        price = forecast_prices[i]
        
        # Calculate changes
        price_change = price - previous_price
        percent_change = (price / previous_price - 1) * 100
        
        print(f"{date_str:<12} | ${price:<14.2f} | ${price_change:<9.2f} | {percent_change:<9.2f}%")
        
        previous_price = price
    
    # Calculate total change from current price
    total_change = forecast_prices[-1] - latest_price
    total_percent = (forecast_prices[-1] / latest_price - 1) * 100
    
    print("-" * 60)
    print(f"Total forecast change: ${total_change:.2f} ({total_percent:.2f}%)")

def main():
    # Parameters
    window_size = 5  # Number of previous days to use for prediction
    forecast_days = 10  # Number of days to forecast
    
    # Step 1: Download data if needed
    if not os.path.exists("MSFT_data.csv"):
        df = download_msft_data(period="5y")
    else:
        print("Loading existing MSFT data...")
        df = pd.read_csv("MSFT_data.csv")
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Step 2: Prepare data
    dates = df['Date'].values
    prices = df['Close'].values
    
    # Create windows for LSTM
    windowed_dates, X, y = df_to_windowed_df(df, window_size)
    
    # Step 3: Split data
    train_size = int(len(X) * 0.8)
    val_size = int(len(X) * 0.1)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Step 4: Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit scaler on training data only
    scaler.fit(y_train.reshape(-1, 1))
    
    # Scale all datasets
    y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    # Also scale X data
    for i in range(len(X_train)):
        for j in range(window_size):
            X_train[i][j] = scaler.transform([[X_train[i][j][0]]])[0][0]
    
    for i in range(len(X_val)):
        for j in range(window_size):
            X_val[i][j] = scaler.transform([[X_val[i][j][0]]])[0][0]
            
    for i in range(len(X_test)):
        for j in range(window_size):
            X_test[i][j] = scaler.transform([[X_test[i][j][0]]])[0][0]
    
    # Step 5: Train or load model
    model_path = 'msft_model.keras'
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        model = train_model(X_train, y_train_scaled, X_val, y_val_scaled, 
                          window_size=window_size, epochs=100, batch_size=32)
        save_model_and_scaler(model, scaler)
    
    # Step 6: Get latest data for prediction
    latest_window = prices[-window_size:].reshape(-1, 1)
    latest_date = dates[-1]
    
    if isinstance(latest_date, np.datetime64):
        latest_date = pd.Timestamp(latest_date).to_pydatetime()
    
    # Step 7: Make forecast
    forecast_prices = predict_n_days_ahead(model, scaler, latest_window, 
                                        n_days=forecast_days, window_size=window_size)
    
    # Step 8: Generate future dates
    future_dates = get_future_dates(latest_date, forecast_days)
    
    # Step 9: Plot results
    plot_predictions(dates, prices, future_dates, forecast_prices, window_size)
    
    # Step 10: Print forecast table
    print_forecast_table(prices[-1], future_dates, forecast_prices)
    
    print(f"\nLatest closing price (on {pd.Timestamp(latest_date).strftime('%Y-%m-%d')}): ${prices[-1]:.2f}")
    print(f"Forecast for {forecast_days} trading days completed!")

if __name__ == "__main__":
    main()