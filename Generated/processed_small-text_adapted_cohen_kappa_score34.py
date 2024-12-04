import numpy as np
from sklearn.metrics import confusion_matrix

def _adapted_cohen_kappa_score(y1, y2, *, labels=None, weights=None, sample_weight=None):
    """
    Calculate Cohen's kappa score with handling for perfect agreement to prevent division by zero error.
    
    Parameters:
    y1 : array-like of shape (n_samples,)
        First set of labels.
    y2 : array-like of shape (n_samples,)
        Second set of labels.
    labels : array-like of shape (n_classes,), default=None
        List of labels to index the matrix. This may be used to reorder or select a subset of labels.
    weights : {'linear', 'quadratic'}, default=None
        Weighting type to calculate the score.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    
    Returns:
    kappa : float
        The kappa statistic, which ranges from -1 to 1.
    """
    if labels is None:
        labels = np.unique(np.concatenate([y1, y2]))
    
    # Compute the confusion matrix
    cm = confusion_matrix(y1, y2, labels=labels, sample_weight=sample_weight)
    
    # Check for perfect agreement
    if np.array_equal(y1, y2):
        return 1.0
    
    n_classes = len(labels)
    sum0 = np.sum(cm, axis=0)
    sum1 = np.sum(cm, axis=1)
    expected = np.outer(sum1, sum0) / np.sum(cm)
    
    if weights is None:
        w_mat = np.ones((n_classes, n_classes), dtype=int)
        w_mat.flat[:: n_classes + 1] = 0
    elif weights == 'linear':
        w_mat = np.abs(np.subtract.outer(np.arange(n_classes), np.arange(n_classes)))
    elif weights == 'quadratic':
        w_mat = np.square(np.subtract.outer(np.arange(n_classes), np.arange(n_classes)))
    else:
        raise ValueError("Unknown kappa weighting type.")
    
    k = np.sum(w_mat * cm) / np.sum(w_mat * expected)
    return 1 - k

