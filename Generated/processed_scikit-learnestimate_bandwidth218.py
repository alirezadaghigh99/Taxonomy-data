import numpy as np
from sklearn.neighbors import NearestNeighbors

def estimate_bandwidth(X, quantile=0.3, n_samples=None, random_state=0, n_jobs=None):
    """
    Estimate the bandwidth to use with the mean-shift algorithm.

    Parameters:
    - X: array-like of shape (n_samples, n_features)
        Input points.
    - quantile: float, default=0.3
        Quantile to use for bandwidth estimation.
    - n_samples: int, default=None
        Number of samples to use. If None, all samples are used.
    - random_state: int, default=0
        Random state for reproducibility.
    - n_jobs: int, default=None
        The number of parallel jobs to run for neighbors search.

    Returns:
    - bandwidth: float
        The estimated bandwidth.
    """
    if n_samples is not None and n_samples < X.shape[0]:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(X.shape[0], n_samples, replace=False)
        X = X[indices]

    nbrs = NearestNeighbors(n_neighbors=int(X.shape[0] * quantile), n_jobs=n_jobs)
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    bandwidth = np.mean(distances[:, -1])
    
    return bandwidth

