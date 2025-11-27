import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# -----------------------------
# Utility functions
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end)
    data = data.dropna()
    return data

def add_sma(df: pd.DataFrame, windows=(25, 50)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()
    return df

def create_sequences(data: np.ndarray, look_back: int):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back, 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

def build_lstm_model(look_back: int, units: int = 16,
                     dropout: float = 0.3, lr: float = 1e-3) -> Sequential:
    """
    Small, robust LSTM based on your tuned model:
    - 1 LSTM layer
    - Dropout for regularisation
    """
    model = Sequential()
    model.add(
        LSTM(
            units=units,
            return_sequences=False,
            input_shape=(look_back, 1),
        )
    )
    model.add(Dropout(dropout))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model

def forecast_n_days(model, scaler, last_prices, look_back: int, n_days: int):
    """
    Recursive multi-step forecast from the last `look_back` prices (orig scale).
    """
    scaled_seq = scaler.transform(last_prices.reshape(-1, 1)).flatten()
    preds_scaled = []

    for _ in range(n_days):
        inp = scaled_seq.reshape(1, look_back, 1)
        next_scaled = model.predict(inp, verbose=0)[0, 0]
        preds_scaled.append(next_scaled)
        scaled_seq = np.append(scaled_seq[1:], next_scaled)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    return preds

def multi_step_test_predictions(
    model,
    scaler,
    scaled_test_close: np.ndarray,
    test_index: pd.DatetimeIndex,
    look_back: int,
    horizon: int,
):
    """
    Evaluate multi-step performance on the test period.
    horizon: 1, 2, 5, ...
    Returns: np.array(dates), actuals, preds
    """
    data = np.asarray(scaled_test_close).reshape(-1, 1)
    N = len(data)

    preds_scaled = []
    actual_scaled = []
    dates = []

    n_iters = N - look_back - horizon + 1
    for i in range(n_iters):
        seq = data[i : i + look_back, 0].copy()

        # recursive prediction horizon steps ahead
        for _ in range(horizon):
            inp = seq.reshape(1, look_back, 1)
            next_scaled = model.predict(inp, verbose=0)[0, 0]
            seq = np.append(seq[1:], next_scaled)

        preds_scaled.append(next_scaled)
        target_idx = i + look_back + horizon - 1
        actual_scaled.append(data[target_idx, 0])
        dates.append(test_index[target_idx])

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(np.array(actual_scaled).reshape(-1, 1)).flatten()
    return np.array(dates), actuals, preds

def compute_metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    r2 = r2_score(actual, pred)
    return mae, mse, rmse, mape, r2

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="LSTM Stock Forecaster",
    page_icon="📈",
    layout="wide",
)

st.title("📈 LSTM Stock Price Forecasting App")
st.markdown(
    """
This version uses your **tuned LSTM model** (small & robust) to
forecast stock prices and evaluate **1-, 2-, and 5-day ahead** accuracy.
"""
)

# Sidebar controls
st.sidebar.header("Configuration")

default_start = dt.date(2015, 1, 1)
default_end = dt.date.today()

ticker = st.sidebar.text_input("Ticker symbol", value="V")
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=default_end)

look_back = st.sidebar.slider("Look-back window (days)", 20, 120, 60, step=5)
train_split = st.sidebar.slider("Training proportion", 0.6, 0.95, 0.85, step=0.05)
epochs = st.sidebar.slider("Max epochs", 10, 200, 100, step=10)
batch_size = st.sidebar.selectbox("Batch size", [16, 32, 64], index=1)

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# -----------------------------
# Load & show data
# -----------------------------
with st.spinner("Downloading data from Yahoo Finance..."):
    data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("No data returned. Check ticker or date range.")
    st.stop()

st.subheader(f"Raw data for {ticker}")
st.write(data.tail())

# -----------------------------
# Price + SMA plot
# -----------------------------
data_sma = add_sma(data, windows=(25, 50))

st.subheader("Historical Price with 25/50-day Simple Moving Averages")

fig_price, ax_price = plt.subplots(figsize=(10, 4))
ax_price.plot(data_sma.index, data_sma["Close"], label="Close", linewidth=1.5)
if "SMA_25" in data_sma.columns:
    ax_price.plot(data_sma.index, data_sma["SMA_25"], label="SMA 25", linewidth=1)
if "SMA_50" in data_sma.columns:
    ax_price.plot(data_sma.index, data_sma["SMA_50"], label="SMA 50", linewidth=1)

ax_price.set_xlabel("Date")
ax_price.set_ylabel("Price")
ax_price.legend()
ax_price.grid(True)
st.pyplot(fig_price)


# -----------------------------
# Train / Test split & scaling
# -----------------------------
st.subheader("Train LSTM model")

close_prices = data_sma["Close"].values.reshape(-1, 1)
N = len(close_prices)
split_idx = int(N * train_split)

train_close = close_prices[:split_idx]
test_close = close_prices[split_idx:]

train_index = data_sma.index[:split_idx]
test_index = data_sma.index[split_idx:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_close = scaler.fit_transform(train_close)
scaled_test_close = scaler.transform(test_close)

X_train, y_train = create_sequences(scaled_train_close, look_back)
X_test, y_test = create_sequences(scaled_test_close, look_back)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

st.write(f"Training sequences: {X_train.shape}, Test sequences: {X_test.shape}")

# Align test dates with y_test
aligned_test_index = test_index[look_back: look_back + len(y_test)]

# -----------------------------
# Train model (button)
# -----------------------------
if st.button("🚀 Train / Re-train LSTM model"):
    with st.spinner("Training LSTM model..."):  
        model = build_lstm_model(look_back=look_back, units=16, dropout=0.3, lr=1e-3)

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )

        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0,
        )

    st.success("Training complete!")

    # ---- Plot training history ----
    st.markdown("### Training vs Validation Loss")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
    ax_hist.plot(history.history["loss"], label="Training loss")
    ax_hist.plot(history.history["val_loss"], label="Validation loss")
    ax_hist.set_xlabel("Epoch")
    ax_hist.set_ylabel("Loss (MSE)")
    ax_hist.legend()
    ax_hist.grid(True)
st.pyplot(fig_hist)

    # ---- 1-step predictions on test set ----
    preds_1step_scaled = model.predict(X_test)
    preds_1step = scaler.inverse_transform(preds_1step_scaled)
    actual_1step = scaler.inverse_transform(y_test.reshape(-1, 1))

    st.markdown("### Test Set: Actual vs 1-step Ahead Prediction")
    fig_test, ax_test = plt.subplots(figsize=(12, 4))
    ax_test.plot(aligned_test_index, actual_1step, label="Actual", linewidth=1.5)
    ax_test.plot(
        aligned_test_index,
        preds_1step,
        label="1-step Prediction",
        linestyle="--",
        linewidth=1.2,
    )
    ax_test.set_xlabel("Date")
    ax_test.set_ylabel("Price")
    ax_test.legend()
    ax_test.grid(True)
st.pyplot(fig_test)

    # ---- Multi-step metrics ----
    st.markdown("### Multi-step Forecast Performance on Test Set")

    dates_1, actual_1, pred_1 = multi_step_test_predictions(
        model, scaler, scaled_test_close, test_index, look_back, horizon=1
    )
    dates_2, actual_2, pred_2 = multi_step_test_predictions(
        model, scaler, scaled_test_close, test_index, look_back, horizon=2
    )
    dates_5, actual_5, pred_5 = multi_step_test_predictions(
        model, scaler, scaled_test_close, test_index, look_back, horizon=5
    )

    metrics = []
    for horizon, a, p in [(1, actual_1, pred_1),
                          (2, actual_2, pred_2),
                          (5, actual_5, pred_5)]:
        mae, mse, rmse, mape, r2 = compute_metrics(a, p)
        metrics.append(
            {
                "Horizon (days)": horizon,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE (%)": mape,
                "R²": r2,
            }
        )

    st.dataframe(pd.DataFrame(metrics).set_index("Horizon (days)"))

    # ---- Plot multi-step forecasts vs actual ----
    st.markdown("### Test Set: Actual vs 1-, 2-, and 5-Day Ahead Forecasts")

    fig_multi, ax_multi = plt.subplots(figsize=(14, 5))
    ax_multi.plot(dates_1, actual_1, label="Actual", color="black", linewidth=1.5)
    ax_multi.plot(dates_1, pred_1, "--", label="1-Day Ahead", linewidth=1)
    ax_multi.plot(dates_2, pred_2, "--", label="2-Day Ahead", linewidth=1)
    ax_multi.plot(dates_5, pred_5, "--", label="5-Day Ahead", linewidth=1)
    ax_multi.set_xlabel("Date")
    ax_multi.set_ylabel("Price")
    ax_multi.legend()
    ax_multi.grid(True)
st.pyplot(fig_multi)

    # ---- Future forecast (next 5 business days) ----
    st.markdown("### Next 5 Business Days Forecast")

    last_lookback_close = data_sma["Close"].values[-look_back:]
    future_preds = forecast_n_days(
        model, scaler, last_lookback_close, look_back=look_back, n_days=5
    )

    last_date = data_sma.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=6, freq="B")[1:]

    df_future = pd.DataFrame(
        {
            "Date": future_dates,
            "Predicted Close": np.round(future_preds, 2),
        }
    )
    df_future.set_index("Date", inplace=True)
    st.dataframe(df_future)

else:
    st.info("Click **'Train / Re-train LSTM model'** to train the model and see forecasts.")