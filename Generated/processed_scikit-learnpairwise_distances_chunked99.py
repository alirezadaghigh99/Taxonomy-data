import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils import gen_batches

def pairwise_distances_chunked(X, Y=None, reduce_func=None, metric='euclidean', n_jobs=None, working_memory=None, **kwds):
    """
    Generate a distance matrix chunk by chunk with optional reduction.

    Parameters:
    - X: array-like of shape (n_samples_X, n_features)
    - Y: array-like of shape (n_samples_Y, n_features), optional
    - reduce_func: callable, optional
        Function to apply on each chunk of the distance matrix.
    - metric: str or callable, default='euclidean'
    - n_jobs: int or None, optional
    - working_memory: int or None, optional
        The number of rows of the distance matrix to fit in memory at once.
    - **kwds: additional keyword parameters for the metric function

    Yields:
    - chunk: array-like
        Contiguous slice of the distance matrix, optionally processed by reduce_func.
    """
    if Y is None:
        Y = X

    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]

    if working_memory is None:
        working_memory = 1024  # Default to 1GB if not specified

    # Estimate the number of rows that can fit in memory
    chunk_size = max(1, int(working_memory * (1024 ** 2) / (n_samples_Y * X.itemsize)))

    for chunk in gen_batches(n_samples_X, chunk_size):
        X_chunk = X[chunk]
        distances = pairwise_distances(X_chunk, Y, metric=metric, n_jobs=n_jobs, **kwds)
        
        if reduce_func is not None:
            yield reduce_func(distances)
        else:
            yield distances

