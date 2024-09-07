from pypfopt import CLA, risk_models, expected_returns
from pypfopt import objective_functions

def risk_parity(data):
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
    cla = CLA(mu, S)
    cla.add_objective(objective_functions.L2_reg)
    weights = cla.efficient_risk(target_volatility=0.15)
    return cla.clean_weights(), cla.portfolio_performance(verbose=False)