import numpy as np
from sklearn.metrics.cluster import pair_confusion_matrix

def rand_score(labels_true, labels_pred):
    """
    Calculate the Rand Index (RI) between two clusterings.
    
    Parameters:
    labels_true (array-like): Ground truth class labels.
    labels_pred (array-like): Cluster labels to evaluate.
    
    Returns:
    float: Rand Index score between 0.0 and 1.0.
    """
    # Compute the pair confusion matrix
    contingency = pair_confusion_matrix(labels_true, labels_pred)
    
    # Extract the elements of the pair confusion matrix
    tn, fp, fn, tp = contingency.ravel()
    
    # Calculate the numerator and denominator for the Rand Index
    numerator = tp + tn
    denominator = tp + tn + fp + fn
    
    # Handle special cases
    if numerator == denominator or denominator == 0:
        return 1.0
    
    # Calculate the Rand Index
    ri_score = numerator / denominator
    
    return ri_score

