import numpy as np
from sklearn.linear_model import lars_path as sklearn_lars_path

def lars_path(X, y, Xy=None, Gram=None, max_iter=500, alpha_min=0, method='lar', 
              copy_X=True, eps=np.finfo(np.float).eps, copy_Gram=True, verbose=0, 
              return_path=True, return_n_iter=False, positive=False):
    """
    Compute Least Angle Regression or Lasso path using the LARS algorithm.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.

    y : array-like, shape (n_samples,)
        Target values.

    Xy : array-like, shape (n_features,), optional
        Xy = np.dot(X.T, y) that can be precomputed. If None, it will be computed.

    Gram : array-like, shape (n_features, n_features), optional
        Precomputed Gram matrix (X^T X), if None, it will be computed.

    max_iter : int, optional (default=500)
        Maximum number of iterations to perform.

    alpha_min : float, optional (default=0)
        Minimum correlation along the path. It corresponds to the regularization
        parameter alpha parameter in the Lasso.

    method : 'lar' | 'lasso', optional (default='lar')
        Specifies the returned model. Select 'lar' for Least Angle Regression, 'lasso'
        for the Lasso.

    copy_X : bool, optional (default=True)
        If True, X will be copied; else, it may be overwritten.

    eps : float, optional (default=np.finfo(np.float).eps)
        The machine-precision regularization in the computation of the Cholesky
        diagonal factors. Increase this for very ill-conditioned systems.

    copy_Gram : bool, optional (default=True)
        If True, Gram will be copied; else, it may be overwritten.

    verbose : int, optional (default=0)
        Controls output verbosity.

    return_path : bool, optional (default=True)
        If True, returns the entire path; else, returns only the last point.

    return_n_iter : bool, optional (default=False)
        If True, returns the number of iterations along with the other outputs.

    positive : bool, optional (default=False)
        Restrict coefficients to be >= 0. This option is only allowed with method='lasso'.

    Returns
    -------
    alphas : array, shape (n_alphas + 1,)
        Maximum of covariances (in absolute value) at each iteration.

    active : list
        Indices of active variables at the end of the path.

    coefs : array, shape (n_features, n_alphas + 1)
        Coefficients along the path.

    n_iter : int
        Number of iterations run. Returned only if return_n_iter is set to True.

    Examples
    --------
    >>> from sklearn import datasets
    >>> from sklearn.linear_model import lars_path
    >>> X, y = datasets.make_regression(n_samples=100, n_features=10, noise=0.1)
    >>> alphas, active, coefs = lars_path(X, y, method='lasso')
    >>> print("Alphas:", alphas)
    >>> print("Active indices:", active)
    >>> print("Coefficients:", coefs)

    References
    ----------
    Efron, B., Hastie, T., Johnstone, I., Tibshirani, R. (2004). Least Angle Regression.
    Annals of Statistics, 32(2), 407-499.
    """
    alphas, active, coefs, n_iter = sklearn_lars_path(
        X, y, Xy=Xy, Gram=Gram, max_iter=max_iter, alpha_min=alpha_min, method=method,
        copy_X=copy_X, eps=eps, copy_Gram=copy_Gram, verbose=verbose, return_path=return_path,
        return_n_iter=return_n_iter, positive=positive
    )

    if return_n_iter:
        return alphas, active, coefs, n_iter
    else:
        return alphas, active, coefs

