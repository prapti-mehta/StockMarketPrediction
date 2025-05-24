import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Function to evaluate different regression models
def evaluate_models(data):
    x = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor()
    }

    results = []

    for name, model in models.items():
        try:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results.append({"Model": name, "MSE": mse, "R²": r2})
        except Exception as e:
            results.append({"Model": name, "MSE": np.nan, "R²": np.nan})
            st.warning(f"{name} failed: {e}")
    
    return results, models.get("Random Forest")

# Function to optimize portfolio
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

# Streamlit App UI
st.title('📈 Stock Price Prediction and Portfolio Optimization')

# --- Stock Prediction Section ---
st.subheader('🔍 Stock Price Prediction')

ticker = st.text_input("Enter Stock Ticker", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime('2020-01-01'))
end_date = st.date_input("End Date", pd.to_datetime('2025-01-01'))

data = fetch_stock_data(ticker, start_date, end_date)

if data is not None:
    st.markdown('#### Stock Closing Price Over Time')
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label='Close')
    plt.title(f'{ticker} Stock Price')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)

    # Evaluate multiple models
    st.markdown('#### Model Performance Comparison')
    results, best_model = evaluate_models(data)
    results_df = pd.DataFrame(results)
    st.dataframe(results_df.style.highlight_max(axis=0, subset=['R²'], color='lightgreen'))

    # Predict future price
    st.markdown('#### Predict Future Price (Using Random Forest)')
    future_data = pd.DataFrame({
        'Open': [data['Open'].iloc[-1]],
        'High': [data['High'].iloc[-1]],
        'Low': [data['Low'].iloc[-1]],
        'Volume': [data['Volume'].iloc[-1]]
    })

    days_ahead = st.slider("Days Ahead to Predict", 1, 30, 1)
    predicted_price = best_model.predict(future_data)[0]
    random_noise = np.random.uniform(-0.03, 0.03)
    st.write(f"Predicted price after {days_ahead} day(s): ${(predicted_price * (1 + random_noise)):.2f}")

# --- Portfolio Optimization Section ---
st.subheader("📊 Portfolio Optimization")

stocks = st.multiselect("Select Stocks", ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'INFY.NS', 'TCS.NS', 'RELIANCE.NS'])
portfolio_start_date = st.date_input("Portfolio Start Date", pd.to_datetime('2015-01-01'))
portfolio_end_date = st.date_input("Portfolio End Date", pd.to_datetime('2025-01-01'))

if stocks:
    cleaned_weights, performance = optimize_portfolio(stocks, portfolio_start_date, portfolio_end_date)
    if cleaned_weights:
        st.markdown("#### Optimized Portfolio Weights")
        st.write(cleaned_weights)

        st.markdown("#### Portfolio Performance")
        st.write(f"Expected Annual Return: {performance[0]*100:.2f}%")
        st.write(f"Annual Volatility: {performance[1]*100:.2f}%")
        st.write(f"Sharpe Ratio: {performance[2]:.2f}")

        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(cleaned_weights.keys(), cleaned_weights.values())
        plt.title("Optimal Portfolio Allocation")
        plt.ylabel("Proportion")
        st.pyplot(plt)
