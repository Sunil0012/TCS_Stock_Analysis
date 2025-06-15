# TCS Stock Data – Live and Latest

A complete machine learning and deep learning project that analyzes and forecasts the stock prices of Tata Consultancy Services (TCS) using historical data. The project includes EDA, feature engineering, linear regression, LSTM modeling, and performance evaluation.

---

## Executive Summary
This project predicts TCS’s stock prices using machine learning models. It involves data cleaning, trend visualization, feature generation, and two prediction models:
- **Linear Regression**: Baseline model using tabular features.
- **LSTM**: Deep learning model using sequential historical data.

LSTM outperformed linear regression with an MAE of ~55, making it a strong candidate for short-term financial forecasting.

---

## Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Introduction](#-introduction)  
3. [Dataset Overview](#-dataset-overview)  
4. [Data Preprocessing](#-data-preprocessing)  
5. [Exploratory Data Analysis](#-exploratory-data-analysis)  
6. [Feature Engineering](#-feature-engineering)  
7. [Modeling](#-modeling)  
8. [Results and Evaluation](#-results-and-evaluation)  
9. [Conclusion](#-conclusion)  
10. [Future Scope](#-future-scope)  
11. [References](#-references)

---
## Exceutive Summary
- This project predicts TCS’s stock prices using machine learning models. It involves data cleaning, trend visualization, feature generation, and two prediction models:
* Linear Regression: Baseline model.
* LSTM: Deep learning model for time series.

- LSTM outperformed linear regression with an MAE of ~55, making it a strong candidate for real-time stock forecasting.


## Introduction
Stock price prediction is challenging but valuable. TCS was selected due to its stable, high-volume trading history. Modern ML and DL models are used to identify patterns and improve forecasting accuracy.

---

## Aims
- Perform EDA on TCS stock data.
- Create temporal, lag, and technical features.
- Train Linear Regression & LSTM models.
- Evaluate model accuracy using MSE, R², MAE.
- Visualize trends and model outputs.

---

## Technology Stack
- **Language**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras  
- **Environments**: Jupyter Notebook, VS Code  
- **Scaling**: MinMaxScaler  
- **Deep Learning**: LSTM (Keras)

---

## Dataset Overview
- **Source**: Public TCS stock data (2002–2021)
- **Files**:
  - `TCS_stock_history.csv` – Daily stock prices
  - `TCS_stock_action.csv` – Corporate actions
  - `TCS_stock_info.csv` – Metadata

### Columns:
- `Date`: Trading day  
- `Open`, `High`, `Low`, `Close`: Price details  
- `Volume`: Shares traded  
- `Dividends`, `Stock Splits`: Corporate events

---

## Data Preprocessing
- Converted `Date` to datetime and sorted
- Checked for missing values (none found)
- Verified data types
- Outlier inspection via plots
- Applied MinMaxScaler (for LSTM)
- Created lag features like `Prev_Close`

---

## Exploratory Data Analysis
- Time series plots of Close prices
- Volume/dividend/split trends
- Correlation heatmap
- Scatter: Close vs Volume
- Moving Averages (30-day, crossover)
- Price change distribution

---

## Feature Engineering
- Extracted: `Year`, `Month`, `Day`, `Day_of_Week`
- Created `Prev_Close` lag feature
- 7-day and 30-day moving averages
- Buy/Sell crossover signals
- Daily price % change

---

## Modeling

### 🔹 Linear Regression
- **Input**: Open, High, Low, Volume, Prev_Close, Month, Day_of_Week  
- **MSE**: 68.11  
- **R²**: 0.9999  
- Strong baseline with good fit.

### 🔹 LSTM Model
- **Input**: Scaled sequential Close prices  
- **MAE**: ~55.22  
- Better short-term tracking with temporal awareness.

---

## Results and Evaluation

| Model            | MAE      | Complexity | Interpretability |
|------------------|----------|------------|------------------|
| Linear Regression| Moderate | Low        | High             |
| LSTM             | ~55.22   | High       | Medium           |

LSTM provides better forecasting for time series data with consistent trend capture.

---

## Conclusion
This project proves the effectiveness of combining:
- EDA + Feature Engineering  
- Linear Regression as a strong baseline  
- LSTM for improved time-series forecasting  

It establishes a practical foundation for applying ML and DL in stock price prediction workflows.

---

## Future Scope
- Integrate **ARIMA** / **Prophet** models
- Add **news sentiment analysis** (BERT, TextBlob)
- Include **macroeconomic indicators**
- Use **model ensembles** (XGBoost + LSTM)
- Build a **Streamlit dashboard** for real-time prediction
- Generalize to **multiple stock symbols**
- Apply **SHAP / LIME** for explainability

---

## References
- [Yahoo Finance](https://finance.yahoo.com)  
- [NSE India](https://www.nseindia.com)  
- [Pandas](https://pandas.pydata.org/docs/)  
- [Matplotlib](https://matplotlib.org/)  
- [Scikit-learn](https://scikit-learn.org/)  
- [TensorFlow / Keras](https://www.tensorflow.org/api_docs/python/tf/keras)  
- [Jason Brownlee - LSTM Guide](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/)  
- [Project GitHub Reference](https://github.com/Vatshayan/Final-Year-Machine-Learning-Stock-Price-Prediction-Project)

---

> Developed by **Khethavath Sunil Naik** | [IIT Bhilai](https://www.iitbhilai.ac.in)
