import numpy as np
from numpy.linalg import norm, lstsq

def orthogonal_mp(X, y, n_nonzero_coefs=None, tol=None, precompute=False, copy_X=True, return_path=False, return_n_iter=False):
    """
    Orthogonal Matching Pursuit (OMP) algorithm.

    Parameters:
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    n_nonzero_coefs : int, optional
        Desired number of non-zero entries in the solution.
    tol : float, optional
        Maximum norm of the residual.
    precompute : bool, optional
        Whether to precompute Gram matrix.
    copy_X : bool, optional
        Whether to copy X.
    return_path : bool, optional
        Whether to return the coefficient path.
    return_n_iter : bool, optional
        Whether to return the number of iterations.

    Returns:
    coef : ndarray of shape (n_features,) or (n_features, n_targets)
        Coefficients of the OMP solution.
    n_iters : int, optional
        Number of active features across every target.
    """
    if copy_X:
        X = X.copy()
    if y.ndim == 1:
        y = y[:, np.newaxis]
    
    n_samples, n_features = X.shape
    n_targets = y.shape[1]
    
    if n_nonzero_coefs is None and tol is None:
        raise ValueError("Either n_nonzero_coefs or tol must be provided.")
    
    if precompute:
        Gram = X.T @ X
        Xy = X.T @ y
    else:
        Gram = None
        Xy = None
    
    coef = np.zeros((n_features, n_targets))
    residuals = y.copy()
    active_set = []
    n_iters = 0
    
    for k in range(n_features):
        if tol is not None and norm(residuals) ** 2 <= tol:
            break
        if n_nonzero_coefs is not None and len(active_set) >= n_nonzero_coefs:
            break
        
        if precompute:
            correlations = Xy - Gram @ coef
        else:
            correlations = X.T @ residuals
        
        new_idx = np.argmax(np.abs(correlations), axis=0)
        active_set.append(new_idx)
        
        if precompute:
            A = Gram[np.ix_(active_set, active_set)]
            b = Xy[active_set]
        else:
            A = X[:, active_set]
            b = y
        
        coef_active, _, _, _ = lstsq(A, b, rcond=None)
        
        if precompute:
            coef[active_set] = coef_active
        else:
            coef[active_set] = coef_active
        
        residuals = y - X @ coef
        n_iters += 1
    
    if return_path:
        return coef, active_set, n_iters if return_n_iter else coef, active_set
    return (coef, n_iters) if return_n_iter else coef

