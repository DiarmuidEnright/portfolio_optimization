from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from typing import np.ndarray

def reduce_dimensionality(returns: pd.DataFrame, n_components: int = 3) -> np.ndarray:
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(returns)
    return reduced_data