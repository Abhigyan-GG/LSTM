import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Directories ---
os.makedirs('model', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('scalers', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# --- Configuration ---
CONFIG = {
    'window_size': 60,  # Start with smaller window
    'target': 'Close',
    'train_split': 0.8,
    'val_split': 0.1,  # 80% train, 10% validation, 10% test
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 15,
    'learning_rate': 0.001
}

print("🔧 Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# --- Helper Functions ---
def calculate_technical_indicators(df):
    """Calculate technical indicators safely"""
    df = df.copy()
    
    # Simple moving averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Price features
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Price_Change'] = df['Close'].pct_change()
    
    # Volume features
    df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=10).std()
    
    return df

def create_sequences(data, target, window_size):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(target[i])
    return np.array(X), np.array(y)

def calculate_metrics(y_true, y_pred, set_name):
    """Calculate comprehensive metrics"""
    # Flatten arrays if needed
    y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
    y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
    
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    
    # Avoid division by zero
    mask = y_true_flat != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
    else:
        mape = np.inf
    
    # R² score
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # Directional accuracy
    if len(y_true_flat) > 1:
        actual_direction = np.diff(y_true_flat) > 0
        pred_direction = np.diff(y_pred_flat) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    else:
        directional_accuracy = 0
    
    return {
        f'{set_name}_MSE': mse,
        f'{set_name}_RMSE': rmse,
        f'{set_name}_MAE': mae,
        f'{set_name}_R2': r2,
        f'{set_name}_MAPE': mape,
        f'{set_name}_Directional_Accuracy': directional_accuracy
    }

# --- Load and preprocess data ---
print("\n📊 Loading and preprocessing data...")

# Check if file exists
if not os.path.exists('MSFT.csv'):
    print("❌ Error: MSFT.csv not found!")
    print("Please make sure the file exists in the current directory.")
    exit(1)

df = pd.read_csv('MSFT.csv')
print(f"📈 Loaded {len(df)} rows of data")
print(f"🗓️ Date range: {df['Date'].min()} to {df['Date'].max()}")

# Convert date and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Check for required columns
required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"❌ Error: Missing columns: {missing_cols}")
    exit(1)

print("✅ Data loaded successfully")

# --- Feature Engineering ---
print("\n🔧 Engineering features...")
df = calculate_technical_indicators(df)

# Remove rows with NaN values
df_clean = df.dropna()
print(f"📊 After cleaning: {len(df_clean)} rows ({len(df) - len(df_clean)} rows removed)")

# Select features for training
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                  'SMA_5', 'SMA_10', 'High_Low_Ratio', 'Volume_Ratio', 'Volatility']

# Ensure all feature columns exist
available_features = [col for col in feature_columns if col in df_clean.columns]
print(f"🎯 Using features: {available_features}")

X_data = df_clean[available_features].values
y_data = df_clean[CONFIG['target']].values

print(f"📊 Feature data shape: {X_data.shape}")
print(f"🎯 Target data shape: {y_data.shape}")

# --- Scaling ---
print("\n⚖️ Scaling data...")

# Scale features
feature_scaler = MinMaxScaler()
X_scaled = feature_scaler.fit_transform(X_data)

# Scale target
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(y_data.reshape(-1, 1)).flatten()

# Save scalers
joblib.dump(feature_scaler, 'scalers/feature_scaler.pkl')
joblib.dump(target_scaler, 'scalers/target_scaler.pkl')

print("✅ Scaling completed")

# --- Create Sequences ---
print(f"\n🪟 Creating sequences with window size: {CONFIG['window_size']}")
X_seq, y_seq = create_sequences(X_scaled, y_scaled, CONFIG['window_size'])

print(f"📊 Sequence shapes: X={X_seq.shape}, y={y_seq.shape}")

# --- Train/Validation/Test Split ---
train_size = int(CONFIG['train_split'] * len(X_seq))
val_size = int(CONFIG['val_split'] * len(X_seq))

X_train = X_seq[:train_size]
y_train = y_seq[:train_size]
X_val = X_seq[train_size:train_size + val_size]
y_val = y_seq[train_size:train_size + val_size]
X_test = X_seq[train_size + val_size:]
y_test = y_seq[train_size + val_size:]

print(f"\n📊 Data splits:")
print(f"  Train: {len(X_train)} samples")
print(f"  Validation: {len(X_val)} samples")
print(f"  Test: {len(X_test)} samples")

# --- Build Model ---
print("\n🏗️ Building LSTM model...")

def build_lstm_model(input_shape):
    """Build an improved LSTM model"""
    model = models.Sequential([
        # First LSTM layer
        layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        
        # Second LSTM layer
        layers.LSTM(50, return_sequences=False),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(25, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='mse',
        metrics=['mae']
    )
    
    return model

model = build_lstm_model((CONFIG['window_size'], len(available_features)))
print("🎯 Model Summary:")
model.summary()

# --- Training ---
print(f"\n🚀 Training model...")

# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=CONFIG['early_stopping_patience'],
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("✅ Training completed!")

# --- Predictions ---
print("\n🔮 Making predictions...")

y_train_pred_scaled = model.predict(X_train, verbose=0)
y_val_pred_scaled = model.predict(X_val, verbose=0)
y_test_pred_scaled = model.predict(X_test, verbose=0)

# --- Inverse Transform ---
print("🔄 Inverse transforming predictions...")

# Reshape for inverse transform
y_train_pred = target_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1))
y_val_pred = target_scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1))
y_test_pred = target_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1))

y_train_actual = target_scaler.inverse_transform(y_train.reshape(-1, 1))
y_val_actual = target_scaler.inverse_transform(y_val.reshape(-1, 1))
y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))

# --- Calculate Metrics ---
print("\n📊 Calculating metrics...")

train_metrics = calculate_metrics(y_train_actual, y_train_pred, 'Train')
val_metrics = calculate_metrics(y_val_actual, y_val_pred, 'Validation')
test_metrics = calculate_metrics(y_test_actual, y_test_pred, 'Test')

all_metrics = {**train_metrics, **val_metrics, **test_metrics}

# Print metrics
print("\n📈 Model Performance Metrics:")
print("=" * 60)
for metric, value in all_metrics.items():
    if 'MAPE' in metric and value != np.inf:
        print(f"{metric:<25}: {value:.2f}%")
    elif 'Accuracy' in metric:
        print(f"{metric:<25}: {value:.2f}%")
    else:
        print(f"{metric:<25}: {value:.4f}")

# --- Create Results DataFrame ---
print("\n💾 Saving results...")

# Get dates for results
dates = df_clean.index[CONFIG['window_size']:]
train_dates = dates[:len(y_train_actual)]
val_dates = dates[len(y_train_actual):len(y_train_actual) + len(y_val_actual)]
test_dates = dates[len(y_train_actual) + len(y_val_actual):]

# Create comprehensive results
results_df = pd.DataFrame({
    'Date': np.concatenate([train_dates, val_dates, test_dates]),
    'Actual': np.concatenate([y_train_actual.flatten(), y_val_actual.flatten(), y_test_actual.flatten()]),
    'Predicted': np.concatenate([y_train_pred.flatten(), y_val_pred.flatten(), y_test_pred.flatten()]),
    'Set': ['Train'] * len(y_train_actual) + ['Validation'] * len(y_val_actual) + ['Test'] * len(y_test_actual)
})

results_df['Error'] = results_df['Actual'] - results_df['Predicted']
results_df['Absolute_Error'] = np.abs(results_df['Error'])
results_df['Percentage_Error'] = np.abs(results_df['Error'] / results_df['Actual'] * 100)

# Save results
results_df.to_csv('results/improved_predictions.csv', index=False)

# Save metrics
with open('results/improved_metrics.json', 'w') as f:
    # Convert numpy types to native Python types for JSON serialization
    metrics_json = {}
    for k, v in all_metrics.items():
        if isinstance(v, np.floating):
            metrics_json[k] = float(v)
        elif isinstance(v, np.integer):
            metrics_json[k] = int(v)
        else:
            metrics_json[k] = v
    json.dump(metrics_json, f, indent=4)

# Save configuration
with open('results/improved_config.json', 'w') as f:
    json.dump(CONFIG, f, indent=4)

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('results/improved_training_history.csv', index=False)

# Save model
model.save('model/improved_MSFT_model.h5')

print("✅ All results saved successfully!")
print("\nFiles created:")
print("📊 results/improved_predictions.csv - Detailed predictions")
print("📈 results/improved_metrics.json - Performance metrics")
print("⚙️ results/improved_config.json - Model configuration")
print("📉 results/improved_training_history.csv - Training history")
print("🤖 model/improved_MSFT_model.h5 - Trained model")

# --- Final Performance Summary ---
print(f"\n🎯 Final Test Set Performance:")
print("=" * 40)
print(f"RMSE: ${test_metrics['Test_RMSE']:.2f}")
if test_metrics['Test_MAPE'] != np.inf:
    print(f"MAPE: {test_metrics['Test_MAPE']:.2f}%")
else:
    print("MAPE: Unable to calculate (division by zero)")
print(f"R²: {test_metrics['Test_R2']:.4f}")
print(f"Directional Accuracy: {test_metrics['Test_Directional_Accuracy']:.2f}%")

# Performance interpretation
print(f"\n📈 Performance Interpretation:")
if test_metrics['Test_R2'] > 0.7:
    print("🟢 Excellent model performance!")
elif test_metrics['Test_R2'] > 0.5:
    print("🟡 Good model performance")
elif test_metrics['Test_R2'] > 0.2:
    print("🟠 Fair model performance - consider improvements")
else:
    print("🔴 Poor model performance - needs significant improvements")

print(f"\n💡 Recommendations:")
if test_metrics['Test_RMSE'] > 50:
    print("- Consider feature engineering or different model architecture")
if test_metrics['Test_Directional_Accuracy'] < 55:
    print("- Model struggles with predicting price direction")
if test_metrics['Test_R2'] < 0:
    print("- Model performs worse than baseline - review data preprocessing")