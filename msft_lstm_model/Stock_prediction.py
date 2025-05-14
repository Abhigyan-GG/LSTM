import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Helper function to convert string dates to datetime objects
def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)

# Load and prepare the data
def load_and_prepare_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Select only 'Date' and 'Close' columns
    df = df[['Date', 'Close']]
    
    # Convert Date column to datetime objects
    df['Date'] = df['Date'].apply(str_to_datetime)
    
    # Set Date as index
    df.index = df.pop('Date')
    
    return df

# Function to create windowed dataframe from the original data
def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)

    target_date = first_date
    
    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n+1)
        
        if len(df_subset) != n+1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
        if next_week.empty:
            break
            
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
        
        if last_time:
            break
        
        target_date = next_date

        if target_date >= last_date:
            last_time = True
    
    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates
    
    X = np.array(X)
    for i in range(0, n):
        ret_df[f'Target-{n-i}'] = X[:, i]
    
    ret_df['Target'] = Y

    return ret_df

# Function to convert windowed dataframe to model inputs
def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]

    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)

# Function to build the LSTM model
def build_lstm_model(window_size):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(window_size, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Main execution function
def run_msft_stock_prediction(data_file="MSFT.csv", window_size=3):
    # Load and prepare data
    df = load_and_prepare_data(data_file)
    
    # Plot the original stock price data
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'])
    plt.title('Microsoft Stock Close Price History')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('msft_stock_history.png')
    plt.close()
    
    # Create windowed dataframe
    # Adjust these dates based on your data availability
    windowed_df = df_to_windowed_df(df, 
                                    '2021-03-25', 
                                    '2022-03-23', 
                                    n=window_size)
    
    # Convert to LSTM input format
    dates, X, y = windowed_df_to_date_X_y(windowed_df)
    
    # Split into train, validation, and test sets
    q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]
    
    # Visualize the data split
    plt.figure(figsize=(12, 6))
    plt.plot(dates_train, y_train)
    plt.plot(dates_val, y_val)
    plt.plot(dates_test, y_test)
    plt.title('Data Split: Training, Validation, and Test Sets')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend(['Train', 'Validation', 'Test'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data_split.png')
    plt.close()
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit on training data
    scaler.fit(y_train.reshape(-1, 1))
    
    # Scale all datasets
    y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    # Also scale the input data
    for i in range(len(X_train)):
        X_train[i] = scaler.transform(X_train[i])
    
    for i in range(len(X_val)):
        X_val[i] = scaler.transform(X_val[i])
        
    for i in range(len(X_test)):
        X_test[i] = scaler.transform(X_test[i])
    
    # Build and train the model
    model = build_lstm_model(window_size)
    
    # Early stopping to prevent overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train_scaled,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val_scaled),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Make predictions
    train_predictions_scaled = model.predict(X_train)
    val_predictions_scaled = model.predict(X_val)
    test_predictions_scaled = model.predict(X_test)
    
    # Inverse transform to get actual stock prices
    train_predictions = scaler.inverse_transform(train_predictions_scaled).flatten()
    val_predictions = scaler.inverse_transform(val_predictions_scaled).flatten()
    test_predictions = scaler.inverse_transform(test_predictions_scaled).flatten()
    
    # Plot the predictions vs actual values
    plt.figure(figsize=(12, 6))
    plt.plot(dates_train, train_predictions)
    plt.plot(dates_train, y_train)
    plt.plot(dates_val, val_predictions)
    plt.plot(dates_val, y_val)
    plt.plot(dates_test, test_predictions)
    plt.plot(dates_test, y_test)
    plt.title('LSTM Model: Predictions vs Actual Values')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend(['Training Predictions', 
                'Training Observations',
                'Validation Predictions', 
                'Validation Observations',
                'Testing Predictions', 
                'Testing Observations'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png')
    plt.close()
    
    # Calculate performance metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Test set performance
    test_mse = mean_squared_error(y_test, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    print("\nTest Set Performance:")
    print(f"MSE: {test_mse:.2f}")
    print(f"RMSE: {test_rmse:.2f}")
    print(f"MAE: {test_mae:.2f}")
    print(f"R²: {test_r2:.4f}")
    
    # Recursive predictions (forecast multiple steps ahead)
    recursive_predictions = []
    recursive_dates = np.concatenate([dates_val, dates_test])
    
    # Make a copy of the last window from training data
    last_window = deepcopy(X_train[-1])
    
    # For each date in validation and test set, predict recursively
    for _ in range(len(recursive_dates)):
        # Predict the next value using the current window
        next_prediction_scaled = model.predict(np.array([last_window])).flatten()[0]
        
        # Convert to original scale
        next_prediction = scaler.inverse_transform([[next_prediction_scaled]])[0][0]
        
        # Add to our predictions list
        recursive_predictions.append(next_prediction)
        
        # Update the window for next prediction (remove oldest, add new prediction)
        last_window = np.roll(last_window, -1, axis=0)
        last_window[-1] = [[next_prediction_scaled]]
    
    # Plot original predictions and recursive predictions
    plt.figure(figsize=(12, 6))
    plt.plot(dates_train, train_predictions)
    plt.plot(dates_train, y_train)
    plt.plot(dates_val, val_predictions)
    plt.plot(dates_val, y_val)
    plt.plot(dates_test, test_predictions)
    plt.plot(dates_test, y_test)
    plt.plot(recursive_dates, recursive_predictions)
    plt.title('LSTM Model: One-step vs Recursive Multi-step Predictions')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend(['Training Predictions', 
                'Training Observations',
                'Validation Predictions', 
                'Validation Observations',
                'Testing Predictions', 
                'Testing Observations',
                'Recursive Predictions'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('recursive_predictions.png')
    plt.close()
    
    # Calculate recursive prediction performance
    val_test_combined = np.concatenate([y_val, y_test])
    recursive_mse = mean_squared_error(val_test_combined, recursive_predictions)
    recursive_rmse = np.sqrt(recursive_mse)
    recursive_mae = mean_absolute_error(val_test_combined, recursive_predictions)
    recursive_r2 = r2_score(val_test_combined, recursive_predictions)
    
    print("\nRecursive Prediction Performance:")
    print(f"MSE: {recursive_mse:.2f}")
    print(f"RMSE: {recursive_rmse:.2f}")
    print(f"MAE: {recursive_mae:.2f}")
    print(f"R²: {recursive_r2:.4f}")
    
    return model, scaler

# Run the stock prediction model
if __name__ == "__main__":
    model, scaler = run_msft_stock_prediction()
    
    # Save the model for future use
    model.save('msft_lstm_model')
    
    print("\nModel saved successfully. You can now use it for future predictions.")