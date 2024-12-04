import numpy as np
from sklearn.metrics import pair_confusion_matrix as sklearn_pair_confusion_matrix

def pair_confusion_matrix(labels_true, labels_pred):
    """
    Calculate the 2x2 similarity matrix (contingency matrix) between two clusterings.

    Parameters:
    labels_true (array-like): Ground truth class labels.
    labels_pred (array-like): Predicted cluster labels.

    Returns:
    numpy.ndarray: A 2x2 contingency matrix.
    """
    # Ensure inputs are numpy arrays
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    
    # Calculate the pair confusion matrix using sklearn's implementation
    contingency_matrix = sklearn_pair_confusion_matrix(labels_true, labels_pred)
    
    return contingency_matrix

