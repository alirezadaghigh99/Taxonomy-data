import numpy as np
from numpy.linalg import norm, lstsq

def orthogonal_mp(X, y, n_nonzero_coefs=None, tol=None, precompute='auto', copy_X=True, return_path=False, return_n_iter=False):
    X = np.array(X, copy=copy_X)
    y = np.array(y, copy=False)
    
    n_samples, n_features = X.shape
    
    if n_nonzero_coefs is None:
        n_nonzero_coefs = max(1, n_features // 10)
    
    if n_nonzero_coefs > n_features:
        raise ValueError("Number of non-zero coefficients cannot exceed the number of features.")
    
    if y.ndim == 1:
        y = y[:, np.newaxis]
    
    n_targets = y.shape[1]
    
    if precompute == 'auto':
        precompute = n_targets > 1 and n_samples > 500
    
    if precompute:
        Gram = np.dot(X.T, X)
        Xy = np.dot(X.T, y)
    else:
        Gram = None
        Xy = None
    
    coef = np.zeros((n_features, n_targets))
    residuals = y.copy()
    index = []
    coef_path = []
    
    for k in range(n_nonzero_coefs):
        if tol is not None and norm(residuals) ** 2 <= tol:
            break
        
        if precompute:
            correlations = np.dot(X.T, residuals)
        else:
            correlations = np.dot(X.T, residuals)
        
        new_idx = np.argmax(np.abs(correlations), axis=0)
        new_idx = new_idx[0] if n_targets == 1 else new_idx
        
        if new_idx in index:
            break
        
        index.append(new_idx)
        
        if precompute:
            A = Gram[np.ix_(index, index)]
            b = Xy[index]
        else:
            A = X[:, index]
            b = y
        
        if n_targets == 1:
            coef_temp, _, _, _ = lstsq(A, b, rcond=None)
        else:
            coef_temp, _, _, _ = lstsq(A, b, rcond=None)
        
        coef[index, :] = coef_temp
        
        if return_path:
            coef_path.append(coef.copy())
        
        residuals = y - np.dot(X, coef)
    
    if return_path:
        coef_path = np.array(coef_path)
    
    if return_n_iter:
        if return_path:
            return coef, coef_path, k + 1
        else:
            return coef, k + 1
    else:
        if return_path:
            return coef, coef_path
        else:
            return coef

