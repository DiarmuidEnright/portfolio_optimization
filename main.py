from data.data_loader import fetch_data, calculate_returns
from models.mvo import mean_variance_optimization
from models.risk_parity import risk_parity
from models.hrp import hierarchical_risk_parity
from models.pca import reduce_dimensionality
from models.metrics import value_at_risk, conditional_value_at_risk, drawdown
from models.ml_models import train_predictive_model, predict_returns
from utils.performance import compare_portfolios
from utils.rolling_backtest import rolling_backtest
import pandas as pd
from typing import Dict, Tuple

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
start_date = '2018-01-01'
end_date = '2023-01-01'

data = fetch_data(tickers, start_date, end_date)
returns = calculate_returns(data)

weights_mvo, perf_mvo = mean_variance_optimization(data)
weights_rp, perf_rp = risk_parity(data)
weights_hrp, perf_hrp = hierarchical_risk_parity(returns)

weights_df = compare_portfolios(weights_mvo, weights_rp, weights_hrp)
print("Portfolio Weights Comparison:")
print(weights_df)

print("Risk Metrics:")
print("Value at Risk (VaR):", value_at_risk(returns['AAPL']))
print("Conditional VaR:", conditional_value_at_risk(returns['AAPL']))
print("Drawdown:", drawdown(returns['AAPL']).max())

reduced_data = reduce_dimensionality(returns)

model = train_predictive_model(reduced_data, returns['AAPL'])
predictions = predict_returns(model, reduced_data)
print("Predicted Returns:", predictions)

rolling_backtest(returns)