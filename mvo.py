from pypfopt import EfficientFrontier, risk_models, expected_returns

def mean_variance_optimization(data):
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    ef.add_constraint(lambda w: w >= 0.02)
    weights = ef.max_sharpe()
    return ef.clean_weights(), ef.portfolio_performance(verbose=False)