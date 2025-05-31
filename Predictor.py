import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler

# --- Directories ---
os.makedirs('model', exist_ok=True)
os.makedirs('results', exist_ok=True)

# --- Load and preprocess data ---
df = pd.read_csv('MSFT.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
data = df[['Close']]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# --- Windowing ---
def create_sequences(dataset, window_size):
    X, y = [], []
    for i in range(window_size, len(dataset)):
        X.append(dataset[i - window_size:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

window_size = 60
X, y = create_sequences(scaled_data, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))

# --- Split into train and validation ---
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# --- Build LSTM model ---
model = models.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
    layers.LSTM(50),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# --- Train model ---
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# --- Predict ---
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# --- Inverse transform to get actual prices ---
y_train_pred_inv = scaler.inverse_transform(y_train_pred)
y_val_pred_inv = scaler.inverse_transform(y_val_pred)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_val_inv = scaler.inverse_transform(y_val.reshape(-1, 1))

# --- Save model ---
model.save('model/MSFT_model.h5')

# --- Save predictions ---
train_dates = df.index[window_size:split + window_size]
val_dates = df.index[split + window_size:]

results_df = pd.DataFrame({
    'Date': np.concatenate((train_dates, val_dates)),
    'Actual': np.concatenate((y_train_inv.flatten(), y_val_inv.flatten())),
    'Predicted': np.concatenate((y_train_pred_inv.flatten(), y_val_pred_inv.flatten())),
})
results_df['%Error'] = 100 * np.abs((results_df['Actual'] - results_df['Predicted']) / results_df['Actual'])
results_df.to_csv('results/predictions.csv', index=False)
