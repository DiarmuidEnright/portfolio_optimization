from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Tuple, Dict
import matplotlib.pyplot as plt

def hierarchical_risk_parity(returns: pd.DataFrame) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
    correlation_matrix = returns.corr()
    dist_matrix = np.sqrt(0.5 * (1 - correlation_matrix))
    linkage_matrix = linkage(dist_matrix, method='ward')
    
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, labels=returns.columns)
    plt.title('Dendrogram')
    plt.xlabel('Assets')
    plt.ylabel('Distance')
    plt.show()
    
    num_clusters = 4
    clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    
    num_assets = len(returns.columns)
    returns_array = returns.values
    mean_returns = np.mean(returns_array, axis=0)
    cov_matrix = np.cov(returns_array, rowvar=False)
    
    weights = cp.Variable(num_assets)
    returns_exp = cp.sum(cp.multiply(mean_returns, weights))
    risk = cp.quad_form(weights, cov_matrix)
    
    objective = cp.Maximize(returns_exp - 0.5 * risk)
    constraints = [cp.sum(weights) == 1, weights >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    optimal_weights = weights.value
    weights_dict = dict(zip(returns.columns, optimal_weights))
    
    portfolio_return = np.dot(optimal_weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility  # Assuming risk-free rate is 0
    
    return weights_dict, (portfolio_return, portfolio_volatility, sharpe_ratio)

weights, performance = hierarchical_risk_parity(returns)
print(weights)
print(performance)