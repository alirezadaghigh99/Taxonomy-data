import numpy as np
from scipy.linalg import svd, eigh
from sklearn.utils import check_random_state

def _logcosh(x, alpha=1.0):
    return np.tanh(alpha * x), alpha * (1 - np.tanh(alpha * x) ** 2)

def _exp(x):
    return x * np.exp(-x ** 2 / 2), (1 - x ** 2) * np.exp(-x ** 2 / 2)

def _cube(x):
    return x ** 3, 3 * x ** 2

def _whiten(X, whiten, solver):
    if solver == 'svd':
        U, S, V = svd(X, full_matrices=False)
        K = (V.T / S).T
        X_white = np.dot(U, K)
    elif solver == 'eigh':
        cov = np.cov(X, rowvar=False)
        d, E = eigh(cov)
        K = E / np.sqrt(d)
        X_white = np.dot(X, K)
    else:
        raise ValueError("Invalid whitening solver.")
    
    if whiten == 'unit-variance':
        X_white /= np.std(X_white, axis=0)
    elif whiten == 'arbitrary-variance':
        pass
    else:
        raise ValueError("Invalid whitening strategy.")
    
    return X_white, K

def fastica(X, n_components=None, algorithm='parallel', whiten='unit-variance', fun='logcosh', fun_args=None,
            max_iter=200, tol=1e-04, w_init=None, whiten_solver='svd', random_state=None, return_X_mean=False,
            compute_sources=True, return_n_iter=False):
    
    if not isinstance(X, (np.ndarray, list)):
        raise TypeError("X should be array-like.")
    
    X = np.array(X)
    n_samples, n_features = X.shape
    
    if n_components is None:
        n_components = n_features
    
    if fun_args is None:
        fun_args = {}
    
    if fun == 'logcosh':
        g = _logcosh
    elif fun == 'exp':
        g = _exp
    elif fun == 'cube':
        g = _cube
    elif callable(fun):
        g = fun
    else:
        raise ValueError("Invalid function for approximation to neg-entropy.")
    
    random_state = check_random_state(random_state)
    
    X_mean = X.mean(axis=0)
    X -= X_mean
    
    if whiten:
        X_white, K = _whiten(X, whiten, whiten_solver)
    else:
        X_white = X
        K = None
    
    W = np.zeros((n_components, n_components), dtype=X.dtype)
    
    if w_init is None:
        w_init = random_state.normal(size=(n_components, n_components))
    
    W = w_init.copy()
    
    for i in range(max_iter):
        if algorithm == 'parallel':
            gwtx, g_wtx = g(np.dot(W, X_white.T), **fun_args)
            W1 = np.dot(gwtx, X_white) / n_samples - g_wtx.mean(axis=1)[:, np.newaxis] * W
            W1 = np.dot(np.linalg.inv(np.sqrt(np.dot(W1, W1.T))), W1)
        elif algorithm == 'deflation':
            for j in range(n_components):
                w = W[j, :].copy()
                for _ in range(max_iter):
                    gwtx, g_wtx = g(np.dot(w, X_white.T), **fun_args)
                    w1 = (X_white * gwtx).mean(axis=0) - g_wtx.mean() * w
                    w1 /= np.sqrt((w1 ** 2).sum())
                    if np.abs(np.abs((w1 * w).sum()) - 1) < tol:
                        break
                    w = w1
                W[j, :] = w
        else:
            raise ValueError("Invalid algorithm.")
        
        if np.max(np.abs(np.abs(np.diag(np.dot(W, W.T))) - 1)) < tol:
            break
    
    S = np.dot(W, X_white.T).T if compute_sources else None
    
    result = [K, W, S]
    if return_X_mean:
        result.append(X_mean)
    if return_n_iter:
        result.append(i + 1)
    
    return result

