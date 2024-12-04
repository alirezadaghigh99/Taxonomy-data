import numpy as np
from scipy.linalg import qr

def cluster_qr(vectors):
    """
    Finds the discrete partition closest to the eigenvector embedding using QR decomposition.

    Parameters:
    vectors (array-like): An array with shape (n_samples, n_clusters) representing the embedding space of the samples.

    Returns:
    labels (array): An array of integers with shape (n_samples,) representing the cluster labels of the vectors.
    """
    # Perform QR decomposition on the input matrix
    Q, R = qr(vectors)
    
    # Find the index of the maximum absolute value in each row of Q
    labels = np.argmax(np.abs(Q), axis=1)
    
    return labels

