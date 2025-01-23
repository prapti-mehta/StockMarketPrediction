import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return

#fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

#predict stock price
def predict_stock_price(data):
    x = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, model

# Portfolio optimization
def optimize_portfolio(stocks, start_date, end_date):
    try:
        data = yf.download(stocks, start=start_date, end=end_date)
        data.ffill(inplace=True)
        data.dropna(inplace=True)

        adj_close = data['Close']
        returns = adj_close.pct_change().dropna()

        mu = mean_historical_return(adj_close)
        S = CovarianceShrinkage(adj_close).ledoit_wolf()

        effi_fron = EfficientFrontier(mu, S)
        raw_weights = effi_fron.max_sharpe()
        cleaned_weights = effi_fron.clean_weights()

        performance = effi_fron.portfolio_performance(verbose=True)
        return cleaned_weights, performance
    except Exception as e:
        st.error(f"Error during portfolio optimization: {e}")
        return None, None

# Streamlit App
st.title('Stock Price Prediction and Portfolio Optimization')
st.subheader('Stock Price Prediction')

# Input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA)", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime('2020-01-01'))
end_date = st.date_input("End Date", pd.to_datetime('2025-01-01'))

# display stock data
data = fetch_stock_data(ticker, start_date, end_date)
if data is not None:
    st.markdown('<h3 style="font-size: 18px;">Stock Price Over Time</h4>', unsafe_allow_html=True)
    plt.figure(figsize=(10,6))
    plt.plot(data['Close'], label='Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.title(f'{ticker} Stock Price Over Time')
    plt.legend()
    st.pyplot(plt)

    st.markdown(f'<h3 style="font-size: 18px;">Predict {ticker} Stock Price</h4>', unsafe_allow_html=True)
    mse, r2, model = predict_stock_price(data)
    st.write(f"Model Performance: MSE: {mse:.2f}, R²: {r2:.2f}")

    future_data = pd.DataFrame({
        'Open': [data['Open'].iloc[-1]],
        'High': [data['High'].iloc[-1]],
        'Low': [data['Low'].iloc[-1]],
        'Volume': [data['Volume'].iloc[-1]]
    })

    predicted_price = model.predict(future_data)[0]
    days_ahead = st.number_input("Predict Price for X Days Ahead:", min_value=1, max_value=365, value=1)
    st.write(f"Prediction for {days_ahead} day(s) ahead: {predicted_price * (1 + np.random.uniform(-0.05, 0.05)):.2f}")

# Portfolio optimization 
st.subheader("Portfolio Optimization")
stocks = st.multiselect("Select Stocks for Portfolio Optimization", ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'INFY.NS', 'TCS.NS', 'RELIANCE.NS'])
portfolio_start_date = st.date_input("Portfolio Start Date", pd.to_datetime('2015-01-01'))
portfolio_end_date = st.date_input("Portfolio End Date", pd.to_datetime('2025-01-01'))

if len(stocks) > 0:
    cleaned_weights, performance = optimize_portfolio(stocks, portfolio_start_date, portfolio_end_date)
    if cleaned_weights is not None:
        st.write("Optimized Portfolio Weights:")
        st.write(cleaned_weights)

        st.write("Portfolio Performance:")
        st.write(f"Expected annual return: {performance[0]*100:.2f}%")
        st.write(f"Annual volatility: {performance[1]*100:.2f}%")
        st.write(f"Sharpe Ratio: {performance[2]:.2f}")

        # Plotting the optimal portfolio allocation
        st.subheader("Optimal Portfolio Allocation")
        plt.figure(figsize=(10,6))
        plt.bar(cleaned_weights.keys(), cleaned_weights.values())
        plt.title("Optimal Portfolio Allocation")
        plt.ylabel("Proportion")
        plt.xlabel("Stocks")
        st.pyplot(plt)
