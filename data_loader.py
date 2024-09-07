import yfinance as yf
import pandas as pd

def fetch_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data

def calculate_returns(data):
    returns = data.pct_change().dropna()
    returns = returns[(returns < 0.1) & (returns > -0.1)]
    return returns