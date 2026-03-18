# LSTM (MSFT Stock Price Prediction)

This repository contains an LSTM-based stock price forecasting project focused on **Microsoft (MSFT)**. It includes:
- A training script that builds an LSTM model on historical MSFT data (`Predictor.py`)
- A more advanced “train/predict” CLI script that can download fresh data from Yahoo Finance and generate a multi-day forecast (`msft_realtime_prediction.py`)
- A small helper to fetch and save MSFT historical data to CSV (`msft_fetcher.py`)
- A Jupyter notebook experiment (`Stock_predictor.ipynb`)
- A sample dataset (`MSFT.csv`)

> Note: This is an educational project. Stock prediction is noisy and risky—do not treat outputs as financial advice.

---

## Repository layout

- `Predictor.py` — trains an LSTM on `MSFT.csv`, computes metrics, and saves model + artifacts
- `msft_realtime_prediction.py` — downloads data using `yfinance`, engineers many technical indicators, trains or predicts via CLI
- `msft_fetcher.py` — downloads MSFT historical data and writes `MSFT.csv`
- `Stock_predictor.ipynb` — notebook version / experimentation
- `MSFT.csv` — historical MSFT OHLCV dataset (used by `Predictor.py`)
- `model/` — saved Keras model(s)
- `scalers/` — saved scalers (MinMaxScaler pickles)
- `results/` — metrics, predictions, training history outputs
- `__pycache__/` — Python cache (auto-generated)

---

## Features (high-level)

### `Predictor.py` (baseline training pipeline)
- Loads `MSFT.csv` (expects columns: `Date, Open, High, Low, Close, Volume`)
- Adds a small set of technical indicators (e.g., SMA, volume ratios, volatility)
- Uses `MinMaxScaler` for features + target scaling
- Creates windowed sequences (`window_size=60` by default)
- Trains a 2-layer LSTM model with dropout + dense layers
- Saves:
  - `model/improved_MSFT_model.h5`
  - `scalers/feature_scaler.pkl`, `scalers/target_scaler.pkl`
  - `results/improved_predictions.csv`, `results/improved_metrics.json`, `results/improved_training_history.csv`, etc.

### `msft_realtime_prediction.py` (advanced + CLI)
- Can **download** MSFT OHLCV data from Yahoo Finance with retry logic
- Adds many engineered features (moving averages, returns, volatility, RSI, MACD, Bollinger Bands, ADX, ATR, OBV, calendar one-hot features, etc.)
- Supports:
  - `--mode train` to train and save a model
  - `--mode predict` to load a model + scaler and forecast the next N business days
  - Optional sample-data mode if downloads fail
- Produces forecast plot and CSV output (e.g., `msft_latest_forecast.png`, `msft_forecast.csv`)

---

## Requirements

You’ll need Python 3 and common ML/data packages.

Typical dependencies used in this repo:
- `tensorflow`
- `numpy`, `pandas`, `matplotlib`
- `scikit-learn`
- `yfinance`, `requests`
- `statsmodels`
- `ta` (technical analysis indicators)
- `joblib`, `pickle`

### Quick install (example)
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn yfinance requests statsmodels ta joblib
```

---

## Usage

### Option A — Train with the included `MSFT.csv` (baseline script)

1. Ensure `MSFT.csv` exists in the repo root (it is included).
2. Run:
```bash
python Predictor.py
```

Outputs will be written into:
- `model/`
- `scalers/`
- `results/`

---

### Option B — Download data + train/predict via CLI (recommended script)

#### Train a model
```bash
python msft_realtime_prediction.py --mode train --period 5y --window_size 60 --epochs 100
```

This will save:
- a model at `--model_path` (default: `msft_lstm_model`)
- a scaler at `--scaler_path` (default: `msft_scaler.pkl`)

#### Predict next N business days
```bash
python msft_realtime_prediction.py --mode predict --days 10 --period 5y --window_size 60
```

You should see:
- a printed forecast table in the terminal
- forecast plot saved as `msft_latest_forecast.png`
- forecast data saved as `msft_forecast.csv`

#### Use synthetic sample data (if downloads fail / offline)
```bash
python msft_realtime_prediction.py --mode train --use_sample
python msft_realtime_prediction.py --mode predict --use_sample
```

---

## Data

### Fetch/refresh `MSFT.csv`
You can regenerate `MSFT.csv` with:
```bash
python msft_fetcher.py
```

`msft_fetcher.py` is configured for:
- start: `1986-03-13`
- end: `2025-05-31`

---

## Notes / gotchas

- Running training can take time depending on CPU/GPU.
- Forecasting is sensitive to feature engineering, scaling, and window size.
- The “advanced” script saves models in a TensorFlow/Keras format at `--model_path`; make sure you keep the matching scaler file.

---

## License

No license file is currently included. If you want, add a `LICENSE` (MIT/Apache-2.0/etc.) to clarify reuse.

---

## Acknowledgements

- Data source: Yahoo Finance (via `yfinance`)
- Model: LSTM (TensorFlow/Keras)
