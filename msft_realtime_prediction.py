import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import logging
import argparse
import yfinance as yf
import pickle
import os
import time
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
import ta



# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


def download_msft_data(period="5y", interval="1d", max_retries=5, retry_delay=5):
    """Download Microsoft stock data with more history for better training and handle rate limiting"""
    for attempt in range(max_retries):
        try:
            # Create a session with a custom User-Agent to mitigate rate limiting
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            msft = yf.Ticker("MSFT", session=session)
            
            # Use download instead of history for better reliability
            df = yf.download("MSFT", period=period, interval=interval, progress=False)
            df = df.reset_index()
            
            # Check if data is empty
            if df.empty:
                raise ValueError("Retrieved empty dataframe")
                
            # Save to CSV
            df.to_csv("MSFT_data.csv", index=False)
            logging.info(f"Downloaded {len(df)} days of MSFT stock data")
            return df
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logging.warning(f"Attempt {attempt+1} failed: {str(e)}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to fetch MSFT data after {max_retries} attempts: {str(e)}")
                
                # Try to load previously saved data if available
                if os.path.exists("MSFT_data.csv"):
                    logging.info("Loading previously saved MSFT data instead...")
                    try:
                        df = pd.read_csv("MSFT_data.csv")
                        if not df.empty:
                            # Ensure Date column is datetime
                            if 'Date' in df.columns:
                                df['Date'] = pd.to_datetime(df['Date'])
                            logging.info(f"Loaded {len(df)} rows of historical data")
                            return df
                    except Exception as load_ex:
                        logging.error(f"Error loading saved data: {load_ex}")
                
                return None


def add_features(df):
    """Add technical indicators and additional features to improve prediction quality"""
    # Check for required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            logging.error(f"Required column {col} not found in dataframe. Columns: {df.columns}")
            return df
            
    # Create a copy to avoid SettingWithCopyWarning
    result = df.copy()
    
    # Handle any potential missing data
    result = result.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])
    
    # Basic features - with error handling
    try:
        result['MA_5'] = result['Close'].rolling(window=5).mean()
        result['MA_20'] = result['Close'].rolling(window=20).mean()
        result['MA_50'] = result['Close'].rolling(window=50).mean()
        result['MA_200'] = result['Close'].rolling(window=200).mean()
        
        # Volume features
        result['Volume_1d_change'] = result['Volume'].pct_change()
        result['Volume_MA5'] = result['Volume'].rolling(window=5).mean()
        result['Volume_MA20'] = result['Volume'].rolling(window=20).mean()
        
        # Price features
        result['Close_1d_change'] = result['Close'].pct_change()
        result['High_Low_diff'] = result['High'] - result['Low']
        result['Close_Open_diff'] = result['Close'] - result['Open']
        
        # Calculate returns for different time periods
        result['return_1d'] = result['Close'].pct_change(1)
        result['return_5d'] = result['Close'].pct_change(5)
        result['return_10d'] = result['Close'].pct_change(10)
        result['return_20d'] = result['Close'].pct_change(20)
        
        # Volatility measures
        result['volatility_5d'] = result['return_1d'].rolling(window=5).std()
        result['volatility_10d'] = result['return_1d'].rolling(window=10).std()
        result['volatility_20d'] = result['return_1d'].rolling(window=20).std()
        
        # Price position indicators
        result['price_to_MA5'] = result['Close'] / result['MA_5']
        result['price_to_MA20'] = result['Close'] / result['MA_20']
        result['price_to_MA50'] = result['Close'] / result['MA_50']
        
        # Technical indicators using the TA library - with error handling
        try:
            # RSI
            result['RSI'] = ta.momentum.RSIIndicator(result['Close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(result['Close'])
            result['MACD'] = macd.macd()
            result['MACD_signal'] = macd.macd_signal()
            result['MACD_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(result['Close'])
            result['BB_high'] = bollinger.bollinger_hband()
            result['BB_low'] = bollinger.bollinger_lband()
            result['BB_width'] = (result['BB_high'] - result['BB_low']) / result['Close']
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(result['High'], result['Low'], result['Close'])
            result['Stoch_k'] = stoch.stoch()
            result['Stoch_d'] = stoch.stoch_signal()
            
            # ATR - Average True Range (volatility indicator)
            result['ATR'] = ta.volatility.AverageTrueRange(result['High'], result['Low'], result['Close']).average_true_range()
            
            # OBV - On-Balance Volume
            result['OBV'] = ta.volume.OnBalanceVolumeIndicator(result['Close'], result['Volume']).on_balance_volume()
            
            # ADX - Average Directional Index (trend strength)
            adx = ta.trend.ADXIndicator(result['High'], result['Low'], result['Close'])
            result['ADX'] = adx.adx()
            result['DI_pos'] = adx.adx_pos()
            result['DI_neg'] = adx.adx_neg()
            
        except Exception as e:
            logging.warning(f"Error calculating some technical indicators: {e}")
        
        # Day of week (one-hot encoded)
        if 'Date' in result.columns:
            result['DayOfWeek'] = pd.to_datetime(result['Date']).dt.dayofweek
            for i in range(5):  # Trading days only (0-4)
                result[f'Day_{i}'] = (result['DayOfWeek'] == i).astype(int)
                
        # Month of year (can capture seasonality)
        if 'Date' in result.columns:
            result['Month'] = pd.to_datetime(result['Date']).dt.month
            for i in range(1, 13):
                result[f'Month_{i}'] = (result['Month'] == i).astype(int)
                
    except Exception as e:
        logging.error(f"Error adding features: {e}")
        return df
    
    # Drop NaN values created by indicators that use windows
    orig_len = len(result)
    result = result.dropna()
    logging.info(f"Added technical indicators and features. Shape: {result.shape}, dropped {orig_len - len(result)} rows with NaN values")
    
    return result


def check_stationarity(series):
    """Check if a time series is stationary using the Augmented Dickey-Fuller test"""
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    
    if result[1] <= 0.05:
        print("Stationary: The series is likely stationary")
        return True
    else:
        print("Non-Stationary: The series is likely not stationary")
        return False


def prepare_data(df, target_col='Close', sequence_length=60, future_steps=1, test_size=0.2):
    """Prepare data for LSTM model with multiple features"""
    # Select features - excluding Date and non-numeric columns
    feature_columns = [col for col in df.columns if col not in ['Date', 'Dividends', 'Stock Splits', 'DayOfWeek']]
    df_features = df[feature_columns].copy()
    
    # Check stationarity of target column
    is_stationary = check_stationarity(df[target_col])
    
    # Differencing if not stationary (use with caution as it changes interpretation)
    if not is_stationary and False:  # Disabled by default - enable if beneficial
        df_features[f'{target_col}_diff'] = df_features[target_col].diff()
        df_features = df_features.dropna()
        logging.info(f"Applied differencing to make {target_col} stationary")
    
    # Scale data - fit on training portion only
    train_idx = int(len(df_features) * (1 - test_size))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_features[:train_idx])
    scaled_data = scaler.transform(df_features)
    
    # Create sequences
    X, y = [], []
    target_idx = df_features.columns.get_loc(target_col)
    
    for i in range(sequence_length, len(scaled_data) - future_steps + 1):
        X.append(scaled_data[i - sequence_length:i])
        # For multi-step prediction, uncomment below line instead
        # y.append(scaled_data[i:i+future_steps, target_idx])
        y.append(scaled_data[i + future_steps - 1, target_idx])
    
    X, y = np.array(X), np.array(y)
    
    # Split into train and test sets
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logging.info(f"Data prepared: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, df_features.columns.tolist()


def build_advanced_model(input_shape, dropout_rate=0.2):
    """Build an advanced LSTM model with bidirectional layers and more complexity"""
    model = Sequential([
        # First Bidirectional LSTM layer
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Second Bidirectional LSTM layer
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Third LSTM layer
        LSTM(64),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Dense layers
        Dense(32, activation='relu'),
        Dropout(dropout_rate/2),
        Dense(1)
    ])
    
    # Use Adam optimizer with learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.summary()
    return model


def train_model(X_train, y_train, X_test, y_test, model_path='msft_lstm_model', batch_size=32, epochs=100):
    """Train LSTM model with callbacks for better convergence"""
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_advanced_model(input_shape)
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001),
        ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss')
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('msft_training_history.png')
    plt.close()
    
    return model


def evaluate_model(model, X_test, y_test, scaler, feature_names, target_col='Close'):
    """Evaluate model performance with multiple metrics"""
    target_idx = feature_names.index(target_col)
    
    # Get predictions
    scaled_predictions = model.predict(X_test)
    
    # Create dummy array for inverse transform
    dummy = np.zeros((len(scaled_predictions), len(feature_names)))
    dummy[:, target_idx] = scaled_predictions.flatten()
    
    # Inverse transform to get actual values
    predictions = scaler.inverse_transform(dummy)[:, target_idx]
    
    # Get actual values
    dummy = np.zeros((len(y_test), len(feature_names)))
    dummy[:, target_idx] = y_test
    actual = scaler.inverse_transform(dummy)[:, target_idx]
    
    # Calculate metrics
    mse = np.mean((predictions - actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actual))
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    
    print(f"\nModel Evaluation Metrics:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Plot predictions vs actual
    plt.figure(figsize=(14, 7))
    plt.plot(actual, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.title('Model Evaluation: Actual vs Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('msft_model_evaluation.png')
    plt.close()
    
    return rmse, mape


def load_model_and_scaler(model_path='msft_lstm_model', scaler_path='msft_scaler.pkl'):
    """Load model and scaler from files"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
        
    model = tf.keras.models.load_model(model_path)

    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
            scaler = scaler_data['scaler']
            feature_names = scaler_data['feature_names']
    else:
        logging.warning("Scaler not found. A new one will be created and fit on training data.")
        scaler = None
        feature_names = None

    return model, scaler, feature_names


def prepare_latest_data(df, scaler, feature_names, sequence_length):
    """Prepare the latest data for prediction"""
    # Add features
    df_features = add_features(df)
    
    # Select only the features we trained on
    df_features = df_features[feature_names]
    
    # Check if we have enough data
    if len(df_features) < sequence_length:
        raise ValueError(f"Not enough data for the specified sequence length: {sequence_length}")
    
    # Scale the data
    scaled_data = scaler.transform(df_features)
    
    # Get the latest sequence
    latest_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, len(feature_names))
    latest_date = df['Date'].iloc[-1]
    if isinstance(latest_date, str):
        latest_date = datetime.datetime.strptime(latest_date, '%Y-%m-%d')
        
    return latest_sequence, latest_date, df['Close'].iloc[-1]


def predict_next_days(model, scaler, input_sequence, feature_names, target_col, n_days=10):
    """Predict stock prices for the next n days with rolling forecasting"""
    target_idx = feature_names.index(target_col)
    predictions = []
    current_sequence = input_sequence.copy()
    
    for _ in range(n_days):
        # Get the next prediction
        pred = model.predict(current_sequence, verbose=0)[0][0]
        predictions.append(pred)
        
        # Update the sequence for the next prediction (rolling forecast)
        # Create a new row with the prediction
        new_row = current_sequence[0, -1:, :].copy()
        new_row[0, 0, target_idx] = pred
        
        # Roll the sequence and add the new prediction
        current_sequence = np.concatenate([current_sequence[:, 1:, :], new_row], axis=1)
    
    # Convert scaled predictions to actual values
    dummy_array = np.zeros((len(predictions), len(feature_names)))
    dummy_array[:, target_idx] = predictions
    actual_predictions = scaler.inverse_transform(dummy_array)[:, target_idx]
    
    return actual_predictions


def get_next_business_days(start_date, n_days):
    """Generate a list of the next n business days"""
    business_days = []
    current_date = start_date
    
    while len(business_days) < n_days:
        current_date = current_date + datetime.timedelta(days=1)
        # Skip weekends (5 = Saturday, 6 = Sunday)
        if current_date.weekday() < 5:
            business_days.append(current_date)
    
    return business_days


def plot_forecast(df, future_dates, forecast_prices):
    """Plot historical prices and forecast"""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(14, 7))
    
    # Plot historical data (last 120 days)
    historical_data = df.tail(120)
    plt.plot(historical_data['Date'], historical_data['Close'], label='Historical Prices')
    
    # Plot forecast
    plt.plot(future_dates, forecast_prices, 'r--o', label='Predicted Prices')
    
    # Highlight the latest close price
    plt.plot(df['Date'].iloc[-1], df['Close'].iloc[-1], 'bo', markersize=8, label='Latest Close Price')
    
    # Add confidence intervals (if available)
    
    # Add title and labels
    plt.title('MSFT Stock Price Forecast', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig('msft_latest_forecast.png')
    plt.close()
    logging.info("Forecast chart saved as 'msft_latest_forecast.png'")


def print_forecast_table(future_dates, forecast_prices):
    """Print a table of forecasted prices"""
    print("\nMSFT Stock Price Forecast:")
    print("-" * 65)
    print(f"{'Date':<12} | {'Predicted Price':<15} | {'Change $':<10} | {'Change %':<10}")
    print("-" * 65)
    
    for i, (date, price) in enumerate(zip(future_dates, forecast_prices)):
        if i == 0:
            change_dollar = "-"
            change_percent = "-"
        else:
            change_dollar = f"${price - forecast_prices[i-1]:.2f}"
            change_percent = f"{((price / forecast_prices[i-1]) - 1) * 100:.2f}%"
            
        print(f"{date.strftime('%Y-%m-%d'):<12} | ${price:<14.2f} | {change_dollar:<10} | {change_percent:<10}")


def save_forecast_to_csv(dates, prices, latest_close, filename="msft_forecast.csv"):
    """Save forecast to CSV with additional information"""
    changes = ["-"]
    changes_pct = ["-"]
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        change_pct = (prices[i] / prices[i-1] - 1) * 100
        changes.append(f"{change:.2f}")
        changes_pct.append(f"{change_pct:.2f}")
    
    # Calculate changes from latest close price
    changes_from_current = [price - latest_close for price in prices]
    changes_pct_from_current = [(price / latest_close - 1) * 100 for price in prices]
    
    df_forecast = pd.DataFrame({
        'Date': dates,
        'Predicted_Close': prices,
        'Daily_Change': changes,
        'Daily_Change_Pct': changes_pct,
        'Change_From_Current': changes_from_current,
        'Change_From_Current_Pct': changes_pct_from_current
    })
    
    df_forecast.to_csv(filename, index=False)
    logging.info(f"Forecast saved to {filename}")


def calculate_potential_return(current_price, forecast_prices, days):
    """Calculate potential returns and risks"""
    final_price = forecast_prices[-1]
    total_return = (final_price / current_price - 1) * 100
    annual_return = ((final_price / current_price) ** (365 / days) - 1) * 100
    
    # Calculate volatility
    daily_returns = [forecast_prices[i] / forecast_prices[i-1] - 1 for i in range(1, len(forecast_prices))]
    volatility = np.std(daily_returns) * 100
    annualized_volatility = volatility * np.sqrt(252)  # 252 trading days in a year
    
    # Calculate Sharpe ratio (assuming risk-free rate of 2%)
    risk_free_rate = 0.02
    daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
    excess_return = np.mean(daily_returns) - daily_risk_free
    sharpe_ratio = (excess_return / np.std(daily_returns)) * np.sqrt(252)
    
    print("\nPotential Return Analysis:")
    print(f"{days}-day Return: {total_return:.2f}%")
    print(f"Annualized Return: {annual_return:.2f}%")
    print(f"Forecast Volatility (daily): {volatility:.2f}%")
    print(f"Annualized Volatility: {annualized_volatility:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Calculate max drawdown
    cumulative_returns = np.cumprod(np.array([1] + [1 + r for r in daily_returns]))
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns / running_max - 1) * 100
    max_drawdown = np.min(drawdown)
    
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")


def train_mode(args):
    """Function to handle model training"""
    logging.info("Starting model training...")
    
    # Download data with retries
    df = download_msft_data(period=args.period, max_retries=args.retries)
    if df is None:
        logging.warning("Failed to download data. Creating sample dataset for training...")
        df = create_sample_dataset()
    
    # Add features
    df_features = add_features(df)
    
    # Prepare data for training
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
        df_features, 
        target_col='Close', 
        sequence_length=args.window_size,
        test_size=0.2
    )
    
    # Train model
    model = train_model(
        X_train, y_train, X_test, y_test, 
        model_path=args.model_path,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    # Evaluate model
    rmse, mape = evaluate_model(model, X_test, y_test, scaler, feature_names)
    
    # Save the scaler with feature names
    with open(args.scaler_path, 'wb') as f:
        pickle.dump({'scaler': scaler, 'feature_names': feature_names}, f)
    
    logging.info(f"Model trained and saved to {args.model_path}")
    logging.info(f"Scaler saved to {args.scaler_path}")
    logging.info(f"Final RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")


def predict_mode(args):
    """Function to handle prediction using a trained model"""
    logging.info("Starting prediction mode...")
    
    # Download latest data with retries
    df = download_msft_data(period=args.period, max_retries=args.retries)
    if df is None:
        logging.warning("Failed to download data. Using sample dataset for prediction...")
        df = create_sample_dataset()
    
    # Load model and scaler
    try:
        model, scaler, feature_names = load_model_and_scaler(
            model_path=args.model_path,
            scaler_path=args.scaler_path
        )
    except FileNotFoundError as e:
        logging.error(e)
        return
    
    # Prepare latest data
    try:
        latest_sequence, latest_date, latest_close = prepare_latest_data(
            df, scaler, feature_names, 
            sequence_length=args.window_size
        )
    except Exception as e:
        logging.error(f"Error preparing latest data: {e}")
        return
    
    # Predict next n days
    forecast_prices = predict_next_days(
        model, scaler, latest_sequence, 
        feature_names, target_col='Close', 
        n_days=args.days
    )
    
    # Generate future dates
    future_dates = get_next_business_days(latest_date, args.days)
    
    # Print and visualize results
    print(f"\nLatest closing price (on {latest_date.strftime('%Y-%m-%d')}): ${latest_close:.2f}")
    print(f"Predicted closing price for {future_dates[0].strftime('%Y-%m-%d')}: ${forecast_prices[0]:.2f}")
    
    plot_forecast(df, future_dates, forecast_prices)
    print_forecast_table(future_dates, forecast_prices)
    save_forecast_to_csv(future_dates, forecast_prices, latest_close)
    calculate_potential_return(latest_close, forecast_prices, args.days)


def create_sample_dataset(filename='sample_msft_data.csv', days=1000):
    """Create a sample dataset when no data can be downloaded"""
    if os.path.exists(filename):
        logging.info(f"Using existing sample data from {filename}")
        return pd.read_csv(filename)
        
    logging.info(f"Creating sample dataset with {days} days of simulated data")
    # Start date
    start_date = datetime.datetime.now() - datetime.timedelta(days=days)
    dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
    
    # Generate synthetic stock data
    np.random.seed(42)  # For reproducibility
    
    # Start with a price and generate a random walk with drift
    price = 300.0  # Starting price
    prices = [price]
    
    # Parameters
    drift = 0.0001  # Upward drift
    volatility = 0.01  # Daily volatility
    
    # Generate Close prices
    for i in range(1, days):
        # Random walk with drift and volatility
        daily_return = np.random.normal(drift, volatility)
        price = price * (1 + daily_return)
        prices.append(price)
    
    # Generate Open, High, Low based on Close
    opens = [price * (1 + np.random.normal(0, 0.005)) for price in prices]
    highs = [max(o, c) * (1 + abs(np.random.normal(0, 0.005))) for o, c in zip(opens, prices)]
    lows = [min(o, c) * (1 - abs(np.random.normal(0, 0.005))) for o, c in zip(opens, prices)]
    
    # Generate volume
    volumes = [int(abs(np.random.normal(20000000, 5000000))) for _ in range(days)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volumes
    })
    
    # Make sure only business days are included
    df['day_of_week'] = df['Date'].dt.dayofweek
    df = df[df['day_of_week'] < 5].drop(columns='day_of_week')
    
    # Save to CSV
    df.to_csv(filename, index=False)
    logging.info(f"Sample dataset created and saved to {filename}")
    
    return df


def main():
    """Main function to parse arguments and execute the appropriate mode"""
    parser = argparse.ArgumentParser(description='MSFT Stock Price Predictor')
    parser.add_argument('--mode', type=str, default='predict', choices=['train', 'predict', 'sample'],
                        help='Mode: train a new model, predict using existing model, or use sample data')
    parser.add_argument('--days', type=int, default=10, help='Number of days to predict')
    parser.add_argument('--period', type=str, default='5y', help='Yahoo Finance period for data download')
    parser.add_argument('--window_size', type=int, default=60, help='Sequence length for LSTM')
    parser.add_argument('--model_path', type=str, default='msft_lstm_model', help='Path to save/load model')
    parser.add_argument('--scaler_path', type=str, default='msft_scaler.pkl', help='Path to save/load scaler')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum epochs for training')
    parser.add_argument('--use_sample', action='store_true', help='Use sample data instead of downloading')
    parser.add_argument('--retries', type=int, default=3, help='Number of download retries')
    
    args = parser.parse_args()
    
    # Check if we should use sample data
    if args.use_sample or args.mode == 'sample':
        logging.info("Using sample data mode")
        if args.mode == 'train':
            # For training on sample data
            df = create_sample_dataset()
            df_features = add_features(df)
            X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
                df_features, 
                target_col='Close', 
                sequence_length=args.window_size,
                test_size=0.2
            )
            
            model = train_model(
                X_train, y_train, X_test, y_test, 
                model_path=args.model_path,
                batch_size=args.batch_size,
                epochs=args.epochs
            )
            
            with open(args.scaler_path, 'wb') as f:
                pickle.dump({'scaler': scaler, 'feature_names': feature_names}, f)
                
        else:
            # For prediction with sample data
            df = create_sample_dataset()
            try:
                model, scaler, feature_names = load_model_and_scaler(
                    model_path=args.model_path,
                    scaler_path=args.scaler_path
                )
                latest_sequence, latest_date, latest_close = prepare_latest_data(
                    df, scaler, feature_names, 
                    sequence_length=args.window_size
                )
                forecast_prices = predict_next_days(
                    model, scaler, latest_sequence, 
                    feature_names, target_col='Close', 
                    n_days=args.days
                )
                future_dates = get_next_business_days(latest_date, args.days)
                plot_forecast(df, future_dates, forecast_prices)
                print_forecast_table(future_dates, forecast_prices)
                save_forecast_to_csv(future_dates, forecast_prices, latest_close)
                calculate_potential_return(latest_close, forecast_prices, args.days)
            except FileNotFoundError:
                logging.error("Model not found. Please train a model first with --mode train")
    else:
        if args.mode == 'train':
            train_mode(args)
        else:
            predict_mode(args)


if __name__ == "__main__":
    main()