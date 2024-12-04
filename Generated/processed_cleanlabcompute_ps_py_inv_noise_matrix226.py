import numpy as np

def compute_ps_py_inv_noise_matrix(labels, noise_matrix):
    """
    Compute ps := P(labels=k), py := P(true_labels=k), and the inverse noise matrix.

    Parameters
    ----------
    labels : np.ndarray
        A discrete vector of noisy labels, i.e. some labels may be erroneous.
        *Format requirements*: for dataset with `K` classes, labels must be in {0,1,...,K-1}.
    
    noise_matrix : np.ndarray
        A conditional probability matrix (of shape (K, K)) of the form P(label=k_s|true_label=k_y) containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of noise_matrix sum to 1.

    Returns
    -------
    ps : np.ndarray
        The probability distribution of the noisy labels.
    
    py : np.ndarray
        The probability distribution of the true labels.
    
    inv_noise_matrix : np.ndarray
        The inverse of the noise matrix.
    """
    # Number of classes
    K = noise_matrix.shape[0]
    
    # Compute ps: P(labels=k)
    ps = np.bincount(labels, minlength=K) / len(labels)
    
    # Compute the inverse noise matrix
    inv_noise_matrix = np.linalg.inv(noise_matrix)
    
    # Compute py: P(true_labels=k)
    py = inv_noise_matrix @ ps
    
    return ps, py, inv_noise_matrix

