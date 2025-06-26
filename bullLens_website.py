# BullLens: Streamlit Web App for TCS Stock Data Analysis and Prediction

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tqdm import tqdm
import yfinance as yf

# --- Page Config ---
st.set_page_config(page_title="BullLens", layout="wide")
st.title("BullLens - TCS Stock Predictor")

# --- Sidebar ---
st.sidebar.image("https://raw.githubusercontent.com/Sunil0012/TCS_Stock_Analysis/main/Screenshot%202025-06-25%20031247.png", width=220)
st.sidebar.markdown("""
<h2 style='color:#0066cc;'>BullLens</h2>
Your data-driven lens into bullish market trends.
""", unsafe_allow_html=True)

# --- Load Dataset ---
df = pd.read_csv("https://raw.githubusercontent.com/Sunil0012/TCS_Stock_Analysis/main/TCS_stock_history.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values("Date", inplace=True)
df['Prev_Close'] = df['Close'].shift(1)
df['Day_of_Week'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df.dropna(inplace=True)

# --- Feature and Target Split ---
features = ['Open', 'High', 'Low', 'Volume', 'Prev_Close', 'Day_of_Week', 'Month']
X = df[features]
y = df['Close']

# --- Linear Regression Model ---
lr_model = LinearRegression()
lr_model.fit(X, y)
df['LR_Predicted'] = lr_model.predict(X)

# --- LSTM Model Preparation ---
X_train = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_lstm = X_train_scaled[:-1].reshape(-1, 1, 1)
y_train_lstm = X_train_scaled[1:]

model = Sequential()
model.add(LSTM(50, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# --- LSTM Training ---
epochs = 20
batch_size = 16
for epoch in tqdm(range(epochs)):
    for i in range(0, len(X_train_lstm), batch_size):
        X_batch = X_train_lstm[i:i + batch_size]
        y_batch = y_train_lstm[i:i + batch_size]
        model.train_on_batch(X_batch, y_batch)

# --- LSTM Prediction ---
lstm_pred_scaled = model.predict(X_train_scaled.reshape(-1, 1, 1)).flatten()
lstm_pred = scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
padding_len = len(df) - len(lstm_pred)
if padding_len >= 0:
    df['LSTM_Predicted'] = np.append([np.nan] * padding_len, lstm_pred)
else:
    df['LSTM_Predicted'] = lstm_pred[:len(df)]

# --- Prediction Input UI ---
st.subheader("Predict TCS Closing Price")
with st.form("predict_form"):
    open_val = st.number_input("Open", value=float(df['Open'].iloc[-1]))
    high_val = st.number_input("High", value=float(df['High'].iloc[-1]))
    low_val = st.number_input("Low", value=float(df['Low'].iloc[-1]))
    volume_val = st.number_input("Volume", value=float(df['Volume'].iloc[-1]))
    prev_close = st.number_input("Previous Close", value=float(df['Close'].iloc[-1]))
    dow = st.slider("Day of Week", min_value=0, max_value=6, value=int(df['Date'].dt.dayofweek.iloc[-1]))
    month = st.slider("Month", min_value=1, max_value=12, value=int(df['Date'].dt.month.iloc[-1]))
    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([[open_val, high_val, low_val, volume_val, prev_close, dow, month]], columns=features)
        prediction_lr = lr_model.predict(input_data)[0]
        st.success(f"📊 Linear Regression Prediction: ₹{prediction_lr:.2f}")

        # Compare with similar close prices from dataset
        similar_rows = df[np.isclose(df['Close'], prediction_lr, atol=5)]
        if not similar_rows.empty:
            st.markdown("**📘 Historical entries with similar Close values:**")
            st.dataframe(similar_rows[['Date', 'Open', 'Close']].tail(5))

# --- Charts ---
st.subheader("📈 TCS Close Price vs Predictions")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Actual', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df['Date'], y=df['LR_Predicted'], name='Linear Regression', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=df['Date'], y=df['LSTM_Predicted'], name='LSTM', line=dict(color='red')))
fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

# --- Real-time Data ---
st.subheader("📡 Real-Time Stock Chart Viewer")
symbol = st.text_input("Enter NSE symbol (e.g., TCS.NS, INFY.NS)", value="TCS.NS")
if symbol:
    live_data = yf.download(tickers=symbol, period="5d", interval="1h")
    if not live_data.empty:
        fig_live = go.Figure()
        fig_live.add_trace(go.Scatter(x=live_data.index, y=live_data['Close'], name=f'{symbol} Close', line=dict(color='green')))
        fig_live.update_layout(title=f"Live Chart for {symbol}", template="plotly_white")
        st.plotly_chart(fig_live, use_container_width=True)
    else:
        st.warning("No data available for this symbol.")

# --- About Section ---
st.subheader("About BullLens")
st.markdown("""
**BullLens** is a stock market analytics dashboard powered by machine learning.

- 📊 Analyze historical and predicted trends of TCS stock.
- 🔮 Get close price forecasts using Linear Regression and LSTM models.
- 🛰 View real-time data from NSE using Yahoo Finance.

Developed with ❤️ by Sunil Naik
""")
