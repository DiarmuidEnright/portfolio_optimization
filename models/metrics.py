import numpy as np
import pandas as pd

def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    return np.percentile(returns, (1 - confidence_level) * 100)

def conditional_value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    var = value_at_risk(returns, confidence_level)
    return returns[returns <= var].mean()

def drawdown(returns: pd.Series) -> pd.Series:
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    return (cumulative_returns - peak) / peak