# BullLens: Streamlit Web App for TCS Stock Data Analysis and Prediction

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime
from tqdm import tqdm

# --- Page Config ---
st.set_page_config(page_title="BullLens", layout="wide")

# --- Load Data ---
df = pd.read_csv("/workspaces/TCS_Stock_Analysis/TCS_stock_history.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values("Date", inplace=True)

# --- Data Preprocessing ---
df['Prev_Close'] = df['Close'].shift(1)
df['Day_of_Week'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df.dropna(inplace=True)

features = ['Open', 'High', 'Low', 'Volume', 'Prev_Close', 'Day_of_Week', 'Month']
X = df[features]
y = df['Close']

# --- Linear Regression ---
lr_model = LinearRegression()
lr_model.fit(X, y)
lr_pred = lr_model.predict(X)
df['LR_Predicted'] = lr_pred

# --- LSTM Preparation ---


# Prepare the data for LSTM
X_train = df['Close'].values.reshape(-1,1)
y_train = df['Close'].shift(-1).dropna().values

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Define the test data
test_ratio = 0.2
test_size = int(len(df) * test_ratio)
test_data = df[-test_size:]
# Prepare the data for prediction
X_test = test_data['Close'].values.reshape(-1, 1)
X_test_scaled = scaler.transform(X_test)
X_test_lstm = X_test_scaled.reshape(-1, 1, 1)

# Reshape the data for LSTM
X_train_lstm = X_train_scaled[:-1].reshape(-1, 1, 1)
y_train_lstm = X_train_scaled[1:]


model = Sequential()
model.add(LSTM(50, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Set the number of epochs and batch size
epochs = 30
batch_size = 15

# Train the model with tqdm progress bar
for epoch in tqdm(range(epochs)):
 for i in range(0, len(X_train_lstm), batch_size):
     X_batch = X_train_lstm[i:i+batch_size]
     y_batch = y_train_lstm[i:i+batch_size]
     model.train_on_batch(X_batch, y_batch)

# Prepare the data for prediction
X_test = test_data['Close'].values.reshape(-1, 1)
X_test_scaled = scaler.transform(X_test)
X_test_lstm = X_test_scaled.reshape(-1, 1, 1)

lstm_predictions = model.predict(X_test_lstm).flatten()

# Pad to match df length
padding_len = len(df) - len(lstm_predictions)
if padding_len >= 0:
    lstm_padded = np.append([np.nan] * padding_len, lstm_predictions)
    df['LSTM_Predicted'] = lstm_padded
else:
    df['LSTM_Predicted'] = lstm_predictions[:len(df)]





# # Inverse transform of the predictions
# lstm_predictions = lstm_predictions.reshape(-1, 1)
# lstm_predictions = scaler.inverse_transform(lstm_predictions)

# df['LSTM_Predicted'] = np.append([np.nan], lstm_predictions.flatten())



# # --- Scale Close prices ---
# scaler = MinMaxScaler()
# close_scaled = scaler.fit_transform(df[['Close']])

# # --- Prepare data for LSTM ---
# X_lstm = close_scaled[:-1].reshape(-1, 1, 1)
# y_lstm = close_scaled[1:]

# # --- Build and train the LSTM model ---
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

# model = Sequential()
# model.add(LSTM(50, input_shape=(1, 1)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train LSTM
# epochs = 30
# batch_size = 15
# for epoch in range(epochs):
#     for i in range(0, len(X_lstm), batch_size):
#         X_batch = X_lstm[i:i+batch_size]
#         y_batch = y_lstm[i:i+batch_size]
#         model.train_on_batch(X_batch, y_batch)

# # --- Predict ---
# X_test_lstm = close_scaled.reshape(-1, 1, 1)
# lstm_pred_scaled = model.predict(X_test_lstm).flatten()
# # After inverse scaling
# # Inverse scale
# lstm_pred = scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()

# # Compute how many NaNs are needed
# padding_len = len(df) - len(lstm_pred)

# # Safely pad
# if padding_len >= 0:
#     lstm_padded = np.append([np.nan] * padding_len, lstm_pred)
#     df['LSTM_Predicted'] = lstm_padded
# else:
#     # Extra safety: trim prediction if somehow longer
#     df['LSTM_Predicted'] = lstm_pred[:len(df)]





# --- Sidebar ---
st.sidebar.image("/workspaces/TCS_Stock_Analysis/Screenshot 2025-06-25 031247.png", width=200)
st.sidebar.title("BullLens")
st.sidebar.markdown("Your data-driven lens into bullish trends 📈")

# --- Tabs ---
tabs = st.tabs(["📊 Live Chart", "🔮 Predictions", "📈 Indicators", "📉 Volatility", "ℹ About"])

# =====================
# 1. Live Chart
# =====================
with tabs[0]:
    st.subheader("TCS Stock Price Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Close Price")
    st.plotly_chart(fig, use_container_width=True)

# =====================
# 2. Predictions
# =====================
with tabs[1]:
    st.subheader("Stock Price Predictions")
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Actual', line=dict(color='blue')))
    fig_pred.add_trace(go.Scatter(x=df['Date'], y=df['LR_Predicted'], name='LR Prediction', line=dict(color='orange')))
    fig_pred.add_trace(go.Scatter(x=df['Date'], y=df['LSTM_Predicted'], name='LSTM Prediction', line=dict(color='red')))
    fig_pred.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_pred, use_container_width=True)

# =====================
# 3. Indicators
# =====================
with tabs[2]:
    st.subheader("Moving Averages & Signals")
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Signal'] = np.where(df['MA30'] > df['MA50'], 1, -1)

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price', line=dict(color='blue')))
    fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['MA30'], name='MA30', line=dict(color='green')))
    fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], name='MA50', line=dict(color='orange')))
    fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['Close'] * df['Signal'], mode='markers', name='Buy/Sell Signal',
                                marker=dict(color='magenta', size=6)))
    fig_ma.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_ma, use_container_width=True)

# =====================
# 4. Volatility
# =====================
with tabs[3]:
    st.subheader("Daily Price Change Distribution")
    df['Daily_Change_%'] = df['Close'].pct_change() * 100
    fig_hist = px.histogram(df.dropna(), x='Daily_Change_%', nbins=100, marginal='box',
                            title='Daily % Change Distribution', color_discrete_sequence=['orange'])
    st.plotly_chart(fig_hist, use_container_width=True)

# =====================
# 5. About
# =====================
with tabs[4]:
    st.subheader("About BullLens")
    st.markdown("""
    *BullLens* is a stock market analytics platform powered by machine learning.
    This app uses data from Tata Consultancy Services (TCS) to analyze trends and predict stock movements.

    *Tech Stack:*
    - Streamlit (Frontend)
    - Scikit-learn, TensorFlow (Backend ML models)
    - Plotly (Interactive visualizations)

    *Models Used:*
    - Linear Regression for baseline forecasting
    - LSTM (Long Short-Term Memory) for sequential pattern prediction
    
    Created with ❤ by Sunil Naik
    """)
