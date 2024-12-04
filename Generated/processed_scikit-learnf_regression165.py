import numpy as np
from scipy import stats
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot

def r_regression(X, y, center=True):
    """Compute the correlation between each feature in X and the target y."""
    X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
    y = np.asarray(y)
    
    if center:
        X = X - X.mean(axis=0)
        y = y - y.mean()
    
    corr = safe_sparse_dot(X.T, y) / np.sqrt(safe_sparse_dot(X.T, X).diagonal() * np.dot(y, y))
    return corr

def f_regression(X, y, center=True, force_finite=True):
    """Perform univariate linear regression tests and return F-statistic and p-values."""
    X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
    y = np.asarray(y)
    
    n_samples, n_features = X.shape
    
    corr = r_regression(X, y, center=center)
    corr_squared = corr ** 2
    
    # Calculate F-statistic
    f_statistic = corr_squared * (n_samples - 2) / (1 - corr_squared)
    
    # Handle non-finite F-statistics
    if force_finite:
        f_statistic = np.where(np.isfinite(f_statistic), f_statistic, np.inf)
    
    # Calculate p-values
    p_values = stats.f.sf(f_statistic, 1, n_samples - 2)
    
    return f_statistic, p_values

