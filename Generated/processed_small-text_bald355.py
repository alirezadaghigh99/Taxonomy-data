import numpy as np

def _bald(p, eps=1e-8):
    """
    Calculate the Bayesian Active Learning by Disagreement (BALD) score.

    Parameters:
    p (numpy.ndarray): A matrix of predictions with shape (n_samples, n_classes).
    eps (float): A small value to avoid log(0). Default is 1e-8.

    Returns:
    numpy.ndarray: The BALD scores for each sample.
    """
    # Ensure p is a numpy array
    p = np.asarray(p)
    
    # Calculate the expected entropy
    expected_entropy = -np.sum(p * np.log(p + eps), axis=1)
    
    # Calculate the entropy of the expected predictions
    mean_p = np.mean(p, axis=0)
    entropy_of_mean = -np.sum(mean_p * np.log(mean_p + eps))
    
    # Calculate the BALD score
    bald_score = entropy_of_mean - expected_entropy
    
    return bald_score

