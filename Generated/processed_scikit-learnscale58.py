import numpy as np
import scipy.sparse as sp

def scale(X, axis=0, with_mean=True, with_std=True, copy=True):
    """
    Standardize a dataset along any axis.

    Center to the mean and component wise scale to unit variance.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to center and scale.

    axis : {0, 1}, default=0
        Axis used to compute the means and standard deviations along. If 0,
        independently standardize each feature, otherwise (if 1) standardize
        each sample.

    with_mean : bool, default=True
        If True, center the data before scaling.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : bool, default=True
        If False, try to avoid a copy and scale in place.
        This is not guaranteed to always work in place; e.g. if the data is
        a numpy array with an int dtype, a copy will be returned even with
        copy=False.

    Returns
    -------
    X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The transformed data.
    """
    if copy:
        X = X.copy()

    if sp.issparse(X):
        if with_mean:
            raise ValueError("Cannot center sparse matrices: pass `with_mean=False` instead.")
        if with_std:
            mean = np.zeros(X.shape[axis])
            std = np.sqrt(X.multiply(X).mean(axis=axis) - mean ** 2)
            std = np.asarray(std).flatten()
            std[std == 0] = 1
            if axis == 0:
                X = X.multiply(sp.diags(1 / std))
            else:
                X = sp.diags(1 / std).dot(X)
        return X

    if with_mean:
        mean = np.mean(X, axis=axis, keepdims=True)
        X -= mean

    if with_std:
        std = np.std(X, axis=axis, keepdims=True)
        std[std == 0] = 1
        X /= std

    return X

