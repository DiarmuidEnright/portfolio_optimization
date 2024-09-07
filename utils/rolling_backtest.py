from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
from typing import Optional

def rolling_backtest(returns: pd.DataFrame, window_size: int = 252) -> None:
    rolling_windows = returns.rolling(window=window_size)
    for window in rolling_windows:
        if window.shape[0] == window_size:
            mu_rolling = expected_returns.mean_historical_return(window)
            S_rolling = risk_models.CovarianceShrinkage(window).ledoit_wolf()
            ef_rolling = EfficientFrontier(mu_rolling, S_rolling)
            ef_rolling.add_constraint(lambda w: w >= 0.02)
            weights_rolling = ef_rolling.max_sharpe()
            performance_rolling = ef_rolling.portfolio_performance(verbose=False)