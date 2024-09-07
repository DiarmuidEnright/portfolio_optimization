# utils/rolling_backtest.py
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
from typing import Optional

def apply_stress_scenario(returns: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if scenario == 'market_crash':
        return returns * 0.7
    elif scenario == 'interest_rate_hike':
        return returns - 0.05
    elif scenario == 'economic_boost':
        return returns + 0.05
    else:
        raise ValueError("Unknown scenario")

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
            print(f"Rolling Window Performance: {performance_rolling}")

            for scenario in ['market_crash', 'interest_rate_hike', 'economic_boost']:
                stressed_returns = apply_stress_scenario(window, scenario)
                mu_stressed = expected_returns.mean_historical_return(stressed_returns)
                S_stressed = risk_models.CovarianceShrinkage(stressed_returns).ledoit_wolf()
                ef_stressed = EfficientFrontier(mu_stressed, S_stressed)
                ef_stressed.add_constraint(lambda w: w >= 0.02)
                weights_stressed = ef_stressed.max_sharpe()
                performance_stressed = ef_stressed.portfolio_performance(verbose=False)
                print(f"Scenario '{scenario}' Performance: {performance_stressed}")
