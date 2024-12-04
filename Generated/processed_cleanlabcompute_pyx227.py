import numpy as np

def compute_pyx(pred_probs, noise_matrix, inverse_noise_matrix):
    """
    Compute P(true_label=k|x) from P(label=k|x), noise_matrix, and inverse_noise_matrix.

    Parameters
    ----------
    pred_probs : np.ndarray
        P(label=k|x) is a (N x K) matrix with K model-predicted probabilities.
        Each row of this matrix corresponds to an example `x` and contains the model-predicted
        probabilities that `x` belongs to each possible class.
        The columns must be ordered such that these probabilities correspond to class 0,1,2,...
        pred_probs should have been computed using 3 (or higher) fold cross-validation.

    noise_matrix : np.ndarray
        A conditional probability matrix (of shape (K, K)) of the form P(label=k_s|true_label=k_y) containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of noise_matrix sum to 1.

    inverse_noise_matrix : np.ndarray
        A conditional probability matrix (of shape (K, K)) of the form P(true_label=k_y|label=k_s) representing
        the estimated fraction observed examples in each class k_s, that are
        mislabeled examples from every other class k_y. If None, the
        inverse_noise_matrix will be computed from pred_probs and labels.
        Assumes columns of inverse_noise_matrix sum to 1.

    Returns
    -------
    pyx : np.ndarray
        P(true_label=k|x) is a (N, K) matrix of model-predicted probabilities.
        Each row of this matrix corresponds to an example `x` and contains the model-predicted
        probabilities that `x` belongs to each possible class.
        The columns must be ordered such that these probabilities correspond to class 0,1,2,...
        pred_probs should have been computed using 3 (or higher) fold cross-validation.
    """
    
    # Check the shape of pred_probs
    if len(np.shape(pred_probs)) != 2:
        raise ValueError(
            "Input parameter np.ndarray 'pred_probs' has shape "
            + str(np.shape(pred_probs))
            + ", but shape should be (N, K)"
        )
    
    # Check the shape of noise_matrix and inverse_noise_matrix
    if noise_matrix.shape != inverse_noise_matrix.shape:
        raise ValueError("noise_matrix and inverse_noise_matrix must have the same shape")
    
    # Check if the columns of noise_matrix and inverse_noise_matrix sum to 1
    if not np.allclose(np.sum(noise_matrix, axis=0), 1):
        raise ValueError("Columns of noise_matrix must sum to 1")
    if not np.allclose(np.sum(inverse_noise_matrix, axis=0), 1):
        raise ValueError("Columns of inverse_noise_matrix must sum to 1")
    
    # Compute P(true_label=k|x) using the inverse noise matrix
    pyx = np.dot(pred_probs, inverse_noise_matrix)
    
    return pyx