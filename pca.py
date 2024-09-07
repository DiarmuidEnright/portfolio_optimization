from sklearn.decomposition import PCA

def reduce_dimensionality(returns, n_components=3):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(returns)
    return reduced_data