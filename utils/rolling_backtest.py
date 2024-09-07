from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
import numpy as np
from typing import Tuple, List

def apply_stress_scenario(returns: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if scenario == 'market_crash':
        return returns * 0.7
    elif scenario == 'interest_rate_hike':
        return returns - 0.05
    elif scenario == 'economic_boost':
        return returns + 0.05
    else:
        raise ValueError("Unknown scenario")

def identify_risk_stocks(returns: pd.DataFrame) -> Tuple[List[str], List[str]]:
    volatility = returns.std()
    median_volatility = volatility.median()
    low_risk_stocks = volatility[volatility <= median_volatility].index.tolist()
    high_risk_stocks = volatility[volatility > median_volatility].index.tolist()
    return low_risk_stocks, high_risk_stocks

def least_correlated_assets(returns: pd.DataFrame, num_assets: int = 5) -> List[str]:
    correlation_matrix = returns.corr().abs()
    np.fill_diagonal(correlation_matrix.values, 0)
    least_correlated = correlation_matrix.sum().nsmallest(num_assets).index.tolist()
    return least_correlated

def tangency_portfolio(returns: pd.DataFrame) -> Tuple[pd.Series, float]:
    mu = expected_returns.mean_historical_return(returns)
    S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    performance = ef.portfolio_performance(verbose=False)
    return weights, performance

def portfolio_of_portfolios(low_risk_weights: pd.Series, high_risk_weights: pd.Series, alpha: float) -> pd.Series:
    combined_weights = alpha * low_risk_weights.add(high_risk_weights, fill_value=0)
    return combined_weights / combined_weights.sum()

def higher_level_tangency_portfolio(low_risk_weights: pd.Series, high_risk_weights: pd.Series, least_corr_weights: pd.Series) -> pd.Series:
    combined_returns = pd.DataFrame({
        'Low_Risk': (1 + low_risk_weights.index).mean(axis=1),
        'High_Risk': (1 + high_risk_weights.index).mean(axis=1),
        'Least_Correlated': (1 + least_corr_weights.index).mean(axis=1)
    })
    mu_combined = expected_returns.mean_historical_return(combined_returns)
    S_combined = risk_models.CovarianceShrinkage(combined_returns).ledoit_wolf()
    ef_combined = EfficientFrontier(mu_combined, S_combined)
    weights_combined = ef_combined.max_sharpe()
    return weights_combined

def rolling_backtest(returns: pd.DataFrame, window_size: int = 252) -> None:
    rolling_windows = returns.rolling(window=window_size)
    for window in rolling_windows:
        if window.shape[0] == window_size:
            low_risk_stocks, high_risk_stocks = identify_risk_stocks(window)
            low_risk_weights, _ = tangency_portfolio(window[low_risk_stocks])
            high_risk_weights, _ = tangency_portfolio(window[high_risk_stocks])
            
            least_correlated = least_correlated_assets(window)
            least_corr_weights, _ = tangency_portfolio(window[least_correlated])
            
            combined_weights = portfolio_of_portfolios(low_risk_weights, high_risk_weights, alpha=0.5)
            higher_level_weights = higher_level_tangency_portfolio(low_risk_weights, high_risk_weights, least_corr_weights)
            
            print(f"Rolling Window Low-Risk Tangency Portfolio Weights: {low_risk_weights}")
            print(f"Rolling Window High-Risk Tangency Portfolio Weights: {high_risk_weights}")
            print(f"Rolling Window Least Correlated Tangency Portfolio Weights: {least_corr_weights}")
            print(f"Rolling Window Combined Portfolio Weights: {combined_weights}")
            print(f"Rolling Window Higher-Level Tangency Portfolio Weights: {higher_level_weights}")

            for scenario in ['market_crash', 'interest_rate_hike', 'economic_boost']:
                stressed_returns = apply_stress_scenario(window, scenario)
                low_risk_stocks, high_risk_stocks = identify_risk_stocks(stressed_returns)
                least_correlated = least_correlated_assets(stressed_returns)
                low_risk_weights, _ = tangency_portfolio(stressed_returns[low_risk_stocks])
                high_risk_weights, _ = tangency_portfolio(stressed_returns[high_risk_stocks])
                least_corr_weights, _ = tangency_portfolio(stressed_returns[least_correlated])
                
                combined_weights = portfolio_of_portfolios(low_risk_weights, high_risk_weights, alpha=0.5)
                higher_level_weights = higher_level_tangency_portfolio(low_risk_weights, high_risk_weights, least_corr_weights)
                
                print(f"Scenario '{scenario}' Low-Risk Tangency Portfolio Weights: {low_risk_weights}")
                print(f"Scenario '{scenario}' High-Risk Tangency Portfolio Weights: {high_risk_weights}")
                print(f"Scenario '{scenario}' Least Correlated Tangency Portfolio Weights: {least_corr_weights}")
                print(f"Scenario '{scenario}' Combined Portfolio Weights: {combined_weights}")
                print(f"Scenario '{scenario}' Higher-Level Tangency Portfolio Weights: {higher_level_weights}")