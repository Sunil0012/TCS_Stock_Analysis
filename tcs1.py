import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
import datetime

# Load data
history_df = pd.read_csv(r"C:\Users\sunil\Downloads\TCS_stock_history.csv")
action_df = pd.read_csv(r"C:\Users\sunil\Downloads\TCS_stock_action.csv")
info_df = pd.read_csv(r"C:\Users\sunil\Downloads\TCS_stock_info.csv")

# Preprocess date and sort
history_df['Date'] = pd.to_datetime(history_df['Date'])
history_df.sort_values('Date', inplace=True)
history_df.set_index('Date', inplace=True)

# Drop missing values
history_df.dropna(inplace=True)

# Feature scaling for LSTM
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(history_df[['Close']])

# Prepare data for LSTM
look_back = 60
X_lstm, y_lstm = [], []
for i in range(look_back, len(scaled_data)):
    X_lstm.append(scaled_data[i-look_back:i, 0])
    y_lstm.append(scaled_data[i, 0])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

# Build LSTM model
lstm_model = Sequential([
    Input(shape=(X_lstm.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_lstm, y_lstm, epochs=5, batch_size=32, verbose=0)

# Train linear regression model
features = history_df[['Open', 'High', 'Low']]
target = history_df['Close']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.set_page_config(page_title="TCS Stock Closing Price Predictor", layout="wide")
st.markdown("""
    <style>
        .main {
            background-color: #f4f9fc;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            color: #003366;
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .section {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
            margin-bottom: 2rem;
        }
        .label {
            font-weight: bold;
            color: #004080;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>📈 TCS Stock Market Predictor</div>", unsafe_allow_html=True)

st.markdown("""
This platform predicts **future closing prices** of TCS stock using both traditional ML and **LSTM-based sequence models**.
Also includes **company profile**, **stock actions**, **news sentiment** (mock), and **macro indicators**.
""")

# Interactive filtering
st.sidebar.header("📂 Filter Stock History")
date_range = st.sidebar.date_input("Select Date Range:", [history_df.index.min(), history_df.index.max()])
filtered_df = history_df.loc[date_range[0]:date_range[1]]

# Company info tree
st.subheader("🌳 Company Profile Tree")
with st.expander("🔍 Click to View TCS Company Information"):
    for col in info_df.columns:
        st.markdown(f"**{col}:** {info_df[col].iloc[0]}")

# Stock actions
st.subheader("🧾 Stock Actions Per Year")
st.dataframe(action_df, use_container_width=True)

# Historical chart
st.subheader("📊 Historical Stock Data")
st.line_chart(filtered_df[['Close']])

# Prediction Section
st.subheader("🔮 Predict Closing Price (Linear Model)")
col1, col2, col3 = st.columns(3)
with col1:
    open_val = st.number_input("Enter Open Price:", value=float(features['Open'].mean()))
with col2:
    high_val = st.number_input("Enter High Price:", value=float(features['High'].mean()))
with col3:
    low_val = st.number_input("Enter Low Price:", value=float(features['Low'].mean()))

if st.button("Predict with Linear Regression", type="primary"):
    pred_close = model.predict(np.array([[open_val, high_val, low_val]]))[0]
    st.success(f"Predicted Closing Price: ₹{pred_close:.2f}")

# LSTM Prediction
st.subheader("📈 Predict Closing Price (LSTM Model)")
if st.button("Predict Next Closing (LSTM)"):
    recent = scaled_data[-look_back:]
    lstm_input = np.reshape(recent, (1, look_back, 1))
    lstm_pred = lstm_model.predict(lstm_input)[0][0]
    next_close = scaler.inverse_transform([[lstm_pred]])[0][0]
    st.success(f"LSTM Predicted Next Closing Price: ₹{next_close:.2f}")

# Model performance
st.subheader("📉 Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("Mean Squared Error (LR)", f"{mse:.2f}")
with col2:
    st.metric("R² Score (LR)", f"{r2:.2f}")

# Sentiment & Macro indicators (mocked)
st.subheader("📰 News Sentiment & Macro Indicators")
st.info("Sentiment Analysis: 📊 Neutral to Positive based on recent news")
st.success("Macro View: 📈 Indian IT sector shows strong fundamentals for FY25")

# Footer
st.markdown("---")
st.markdown("<center>Created with ❤️ using Streamlit + LSTM + Linear Regression</center>", unsafe_allow_html=True)