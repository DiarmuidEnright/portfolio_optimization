from mlfinlab.portfolio_optimization import HRPOpt
from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
from typing import Tuple, Dict

def hierarchical_risk_parity(returns: pd.DataFrame) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
    hrp = HRPOpt(returns)
    weights = hrp.optimize()
    linkage_matrix = hrp.clusters
    dendrogram(linkage_matrix, labels=returns.columns)
    return weights, hrp.portfolio_performance(verbose=False)