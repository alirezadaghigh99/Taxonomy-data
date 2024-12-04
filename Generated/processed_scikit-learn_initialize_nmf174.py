import numpy as np
from sklearn.utils import check_random_state
from sklearn.decomposition import NMF
from sklearn.utils.extmath import randomized_svd

def _initialize_nmf(X, n_components, init=None, eps=1e-6, random_state=None):
    """
    Compute an initial guess for the non-negative rank k matrix approximation for a given data matrix X using NMF initialization algorithms.

    Parameters:
    - X: array-like of shape (n_samples, n_features) - The data matrix to be decomposed.
    - n_components: int - The number of components desired in the approximation.
    - init: {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}, default=None - Method used to initialize the procedure.
    - eps: float, default=1e-6 - Truncate all values less than this in the output to zero.
    - random_state: int, RandomState instance or None, default=None - Used for reproducible results.

    Returns:
    - W: array-like of shape (n_samples, n_components) - Initial guesses for solving X ~= WH.
    - H: array-like of shape (n_components, n_features) - Initial guesses for solving X ~= WH.
    """
    n_samples, n_features = X.shape
    rng = check_random_state(random_state)

    if init is None:
        init = 'random'

    if init == 'random':
        W = np.abs(rng.randn(n_samples, n_components))
        H = np.abs(rng.randn(n_components, n_features))
        W[W < eps] = 0
        H[H < eps] = 0
        return W, H

    elif init in {'nndsvd', 'nndsvda', 'nndsvdar'}:
        U, S, Vt = randomized_svd(X, n_components, random_state=random_state)
        W = np.zeros((n_samples, n_components))
        H = np.zeros((n_components, n_features))

        W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        H[0, :] = np.sqrt(S[0]) * np.abs(Vt[0, :])

        for j in range(1, n_components):
            x, y = U[:, j], Vt[j, :]
            x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
            x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))
            m_p, m_n = np.linalg.norm(x_p) * np.linalg.norm(y_p), np.linalg.norm(x_n) * np.linalg.norm(y_n)

            if m_p > m_n:
                u = x_p / np.linalg.norm(x_p)
                v = y_p / np.linalg.norm(y_p)
                sigma = m_p
            else:
                u = x_n / np.linalg.norm(x_n)
                v = y_n / np.linalg.norm(y_n)
                sigma = m_n

            W[:, j] = np.sqrt(S[j] * sigma) * u
            H[j, :] = np.sqrt(S[j] * sigma) * v

        if init == 'nndsvda':
            W[W < eps] = 0
            H[H < eps] = 0

        elif init == 'nndsvdar':
            W[W < eps] = 0
            H[H < eps] = 0
            W += eps * rng.randn(n_samples, n_components)
            H += eps * rng.randn(n_components, n_features)
            W[W < 0] = 0
            H[H < 0] = 0

        return W, H

    else:
        raise ValueError(f"Invalid init parameter: got {init}. Expected one of {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}.")

