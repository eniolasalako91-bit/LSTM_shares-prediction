import os
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from plotly.subplots import make_subplots
import plotly.graph_objects as go

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except Exception:
        pass

st.set_page_config(page_title="LSTM Stock Forecast Dashboard", layout="wide")

LOOK_BACK = 50
MODELS_DIR = "models"

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR, exist_ok=True)

@st.cache_data
def load_stock_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame | None:
    df = yf.download(ticker, start=start, end=end)
    if df is None or df.empty:
        return None
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['SMA_25'] = df['Close'].rolling(window=25).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df = df.dropna()
    if df.empty:
        return None
    return df

def prepare_sequences(df: pd.DataFrame, look_back: int = LOOK_BACK):
    close = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close)

    X, y = [], []
    for i in range(len(scaled) - look_back):
        X.append(scaled[i:i + look_back])
        y.append(scaled[i + look_back])
    X = np.array(X)
    y = np.array(y)

    if len(X) < 10:
        raise ValueError("Not enough data to create sequences. Try a longer date range.")

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    X_test = X_test.reshape((X_test.shape[0], look_back, 1))

    all_dates = df.index[look_back:]
    train_dates = all_dates[:train_size]
    test_dates = all_dates[train_size:train_size + len(y_test)]

    return X_train, y_train, X_test, y_test, train_dates, test_dates, scaler, scaled

def build_lstm_model(look_back: int = LOOK_BACK) -> Sequential:
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def get_model_path(ticker: str) -> str:
    safe_ticker = ticker.replace("^", "").upper()
    return os.path.join(MODELS_DIR, f"{safe_ticker}_model.h5")

@st.cache_resource
def train_or_load_model(ticker: str,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        epochs: int,
                        batch_size: int):
    model_path = get_model_path(ticker)
    if os.path.exists(model_path):
        model = load_model(model_path)
        history = None
        return model, history

    model = build_lstm_model(LOOK_BACK)
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    model.save(model_path)
    return model, history

def forecast_future(model,
                    scaled_series: np.ndarray,
                    scaler: MinMaxScaler,
                    look_back: int = LOOK_BACK,
                    days_ahead: int = 7) -> np.ndarray:
    last_window = scaled_series[-look_back:]
    preds_scaled = []
    for _ in range(days_ahead):
        x_input = last_window.reshape(1, look_back, 1)
        next_scaled = model.predict(x_input, verbose=0)[0][0]
        preds_scaled.append(next_scaled)
        last_window = np.append(last_window[1:], next_scaled)
    preds = scaler.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)
    ).flatten()
    return preds

def plot_candlestick_with_volume(df: pd.DataFrame, ticker: str):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="OHLC"
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['SMA_25'],
            mode="lines",
            name="SMA 25"
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['SMA_50'],
            mode="lines",
            name="SMA 50"
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name="Volume",
            opacity=0.6
        ),
        row=2,
        col=1
    )
    fig.update_layout(
        title=f"{ticker} OHLC with SMA25 & SMA50",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig

def plot_actual_vs_pred(actual: np.ndarray, pred: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(actual, label="Actual")
    ax.plot(pred, label="Predicted")
    ax.legend()
    ax.set_title("Actual vs Predicted Close Price")
    ax.grid(True)
    st.pyplot(fig)

def plot_forecast(df: pd.DataFrame,
                  scaler: MinMaxScaler,
                  scaled_series: np.ndarray,
                  model,
                  look_back: int,
                  days_list: list[int]):
    last_close = df['Close'].values[-look_back:]
    last_dates = df.index[-look_back:]

    max_days = max(days_list)
    future_prices = forecast_future(model, scaled_series, scaler, look_back, max_days)

    last_date = df.index[-1]
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1),
                                  periods=max_days)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=last_dates,
            y=last_close,
            mode="lines",
            name="Recent Close"
        )
    )

    table_rows = []
    for d in days_list:
        price = float(future_prices[d - 1])
        date = future_dates[d - 1]
        fig.add_trace(
            go.Scatter(
                x=[date],
                y=[price],
                mode="markers+text",
                name=f"{d}-Day Forecast",
                text=[f"{d}D"],
                textposition="top center"
            )
        )
        table_rows.append({"Days Ahead": d, "Date": date, "Forecast Price": price})

    fig.update_layout(
        title="Short-Term Forecast (1, 2, 5 Days)",
        xaxis_title="Date",
        yaxis_title="Price",
        margin=dict(l=0, r=0, t=40, b=0),
    )

    st.subheader("Forecast Chart (1, 2, 5 Days)")
    st.plotly_chart(fig, use_container_width=True)

    forecast_df = pd.DataFrame(table_rows)
    forecast_df['Date'] = forecast_df['Date'].dt.strftime("%Y-%m-%d")

    st.subheader("Forecast Table")
    st.dataframe(forecast_df.style.format({"Forecast Price": "{:.2f}"}))

    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Forecasts as CSV",
        data=csv,
        file_name="forecasts.csv",
        mime="text/csv"
    )

def main():
    st.title("📈 LSTM Stock Forecast Dashboard")

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Ticker", value="MSFT").upper().strip()
    with col2:
        start = st.date_input("Start Date", value=dt.date(2015, 1, 1))
    with col3:
        end = st.date_input("End Date", value=dt.date.today())

    st.sidebar.header("Model Settings")
    epochs = st.sidebar.slider("Epochs", min_value=10, max_value=100, value=30, step=5)
    batch_size = st.sidebar.selectbox("Batch Size", options=[16, 32, 64], index=1)

    run = st.button("Run Model")

    if not run:
        st.info("Configure parameters and click **Run Model**.")
        return

    if start >= end:
        st.error("Start date must be before end date.")
        return

    with st.spinner("Loading data..."):
        df = load_stock_data(ticker, start, end)

    if df is None or df.empty:
        st.error("No data available for the selected range/ticker.")
        return

    st.subheader(f"Raw Data for {ticker}")
    st.dataframe(df.tail())

    st.subheader("Price & Volume Overview")
    fig_candle = plot_candlestick_with_volume(df, ticker)
    st.plotly_chart(fig_candle, use_container_width=True)

    try:
        X_train, y_train, X_test, y_test, train_dates, test_dates, scaler, scaled_series = prepare_sequences(df, LOOK_BACK)
    except ValueError as e:
        st.error(str(e))
        return

    with st.spinner("Training or loading model..."):
        model, history = train_or_load_model(ticker, X_train, y_train, epochs, batch_size)

    if history is not None:
        st.subheader("Training & Validation Loss")
        fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
        ax_loss.plot(history.history["loss"], label="Training Loss")
        ax_loss.plot(history.history["val_loss"], label="Validation Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid(True)
        st.pyplot(fig_loss)

    with st.spinner("Making predictions..."):
        test_pred_scaled = model.predict(X_test, verbose=0)
        test_pred = scaler.inverse_transform(test_pred_scaled)
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    st.subheader("Actual vs Predicted (Test Set)")
    plot_actual_vs_pred(actual, test_pred)

    mae = mean_absolute_error(actual, test_pred)
    mse = mean_squared_error(actual, test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, test_pred)

    st.subheader("Model Performance (Test Set)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{mae:.4f}")
    c2.metric("MSE", f"{mse:.4f}")
    c3.metric("RMSE", f"{rmse:.4f}")
    c4.metric("R²", f"{r2:.4f}")

    pred_df = pd.DataFrame(
        {
            "Date": test_dates.strftime("%Y-%m-%d"),
            "Actual": actual.flatten(),
            "Predicted": test_pred.flatten(),
        }
    )
    st.subheader("Predictions Table (Test Set)")
    st.dataframe(pred_df.tail().style.format({"Actual": "{:.2f}", "Predicted": "{:.2f}"}))

    csv_preds = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Test Predictions as CSV",
        data=csv_preds,
        file_name="test_predictions.csv",
        mime="text/csv"
    )

    st.subheader("Short-Term Forecasts (1, 2, 5 Days)")
    plot_forecast(df, scaler, scaled_series, model, LOOK_BACK, days_list=[1, 2, 5])

if __name__ == "__main__":
    main()
