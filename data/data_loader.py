import yfinance as yf
import pandas as pd
from typing import Tuple

def fetch_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data

def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    returns = data.pct_change().dropna()
    returns = returns[(returns < 0.1) & (returns > -0.1)]
    return returns

    #Forgot to add anything today :3