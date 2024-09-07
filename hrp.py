from mlfinlab.portfolio_optimization import HRPOpt
from scipy.cluster.hierarchy import dendrogram

def hierarchical_risk_parity(returns):
    hrp = HRPOpt(returns)
    weights = hrp.optimize()
    linkage_matrix = hrp.clusters
    dendrogram(linkage_matrix, labels=returns.columns)
    return weights, hrp.portfolio_performance(verbose=False)