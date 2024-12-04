import numpy as np

def johnson_lindenstrauss_min_dim(n_samples, eps):
    """
    Calculate the minimum number of components required to ensure an eps-embedding
    with good probability for the given number of samples, according to the
    Johnson-Lindenstrauss lemma.

    Parameters:
    n_samples (int or array-like): The number of samples.
    eps (float or array-like): The maximum distortion rate.

    Returns:
    int or np.ndarray: The minimal number of components required.
    """
    if np.any(eps <= 0) or np.any(eps >= 1):
        raise ValueError("eps must be between 0 and 1 (exclusive).")
    
    n_samples = np.asarray(n_samples)
    eps = np.asarray(eps)
    
    min_dim = (4 * np.log(n_samples) / (eps ** 2 / 2 - eps ** 3 / 3)).astype(int)
    
    return min_dim

