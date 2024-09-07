from data.data_loader import fetch_data, calculate_returns
from models.mvo import mean_variance_optimization
from models.risk_parity import risk_parity
from models.hrp import hierarchical_risk_parity
from models.pca import reduce_dimensionality
from utils.performance import compare_portfolios
from utils.rolling_backtest import rolling_backtest

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
start_date = '2018-01-01'
end_date = '2023-01-01'

data = fetch_data(tickers, start_date, end_date)
returns = calculate_returns(data)

weights_mvo, perf_mvo = mean_variance_optimization(data)
weights_rp, perf_rp = risk_parity(data)
weights_hrp, perf_hrp = hierarchical_risk_parity(returns)

weights_df = compare_portfolios(weights_mvo, weights_rp, weights_hrp)
print(weights_df)

reduced_data = reduce_dimensionality(returns)

rolling_backtest(returns)