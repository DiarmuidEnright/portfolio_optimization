from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
from typing import Tuple, Dict

def mean_variance_optimization(data: pd.DataFrame) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    ef.add_constraint(lambda w: w >= 0.02)
    weights = ef.max_sharpe()
    return ef.clean_weights(), ef.portfolio_performance(verbose=False)