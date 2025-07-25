import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from datetime import date

st.set_page_config(page_title='LSTM Stock Predictor', page_icon='📈', layout='centered')

# Developer credit on top right
st.markdown(
    "<div style='text-align:right; font-size:14px; color:gray;'>"
    "Developed by <b>Gattu Navaneeth Rao</b> (Stock Prediction)"
    "</div>",
    unsafe_allow_html=True
)

st.title("📈 Stock Price Prediction (LSTM) – Buy / Hold Signal")

st.write("Enter an **NSE symbol with `.NS`** (e.g., `SBIN.NS`, `COCHINSHIP.NS`, `TCS.NS`).")

# --- Sidebar controls ---
st.sidebar.header("Training Settings")
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2020-01-01").date())
end_date = st.sidebar.date_input("End date", value=date.today())
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=5, step=1)
sequence_length = st.sidebar.slider("Sequence length (lookback days)", 10, 120, 60, 5)
batch_size = st.sidebar.selectbox("Batch size", [16, 32, 64], index=1)
buy_threshold = st.sidebar.number_input("Required % upside for BUY", min_value=0.0, max_value=20.0, value=0.5, step=0.1, help="Predicted > last close by this % triggers BUY; else HOLD.")

stock_symbol = st.text_input("Stock Symbol", "SBIN.NS")

def download_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end, progress=False)

def prepare_sequences(scaled_series, seq_len):
    X, y = [], []
    for i in range(seq_len, len(scaled_series)):
        X.append(scaled_series[i-seq_len:i, 0])
        y.append(scaled_series[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

def build_model(seq_len):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_len, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if stock_symbol:
    data = download_data(stock_symbol, start_date, end_date)
    if data.empty or 'Close' not in data.columns:
        st.error("No data found. Check the symbol (remember `.NS`).")
    else:
        st.subheader(f"{stock_symbol} Closing Price")
        st.line_chart(data['Close'])

        # Scale
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

        # Train
        st.write("Training LSTM model…")
        X_train, y_train = prepare_sequences(scaled_data, sequence_length)
        if len(X_train) < 10:
            st.error("Not enough data after lookback window. Use earlier start date or smaller sequence length.")
        else:
            model = build_model(sequence_length)
            model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
            st.success("Training complete.")

            # Predict last N days
            lookahead_points = min(30, len(scaled_data) - sequence_length)
            if lookahead_points <= 0:
                st.error("Not enough data to generate predictions.")
            else:
                test_data = scaled_data[-(sequence_length + lookahead_points):]
                X_test = []
                for i in range(sequence_length, len(test_data)):
                    X_test.append(test_data[i-sequence_length:i, 0])
                X_test = np.array(X_test).reshape(-1, sequence_length, 1)

                predicted_scaled = model.predict(X_test)
                predicted_prices = scaler.inverse_transform(predicted_scaled)

                # Align with actuals
                actual_slice = data['Close'].iloc[-predicted_prices.shape[0]:]
                pred_index = actual_slice.index
                df_plot = pd.DataFrame({
                    'Actual': actual_slice.values.ravel(),
                    'Predicted': predicted_prices.ravel()
                }, index=pred_index)

                st.subheader("Actual vs Predicted (Recent)")
                st.line_chart(df_plot)

                # Next-step signal
                last_real_price = float(data['Close'].iloc[-1])
                predicted_next_price = float(predicted_prices[-1])

                pct_up = ((predicted_next_price - last_real_price) / last_real_price) * 100.0

                if pct_up >= buy_threshold:
                    st.success(f"Predicted ₹{predicted_next_price:,.2f} (+{pct_up:.2f}% vs last ₹{last_real_price:,.2f}). **BUY**.")
                else:
                    st.warning(f"Predicted ₹{predicted_next_price:,.2f} (+{pct_up:.2f}% vs last ₹{last_real_price:,.2f}). **HOLD**.")

                st.caption("Educational demo only. Not investment advice.")
