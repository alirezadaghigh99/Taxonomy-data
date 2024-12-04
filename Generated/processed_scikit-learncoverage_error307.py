import numpy as np

def coverage_error(y_true, y_score, sample_weight=None):
    """
    Compute the coverage error measure.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_labels)
        True binary labels in binary indicator format.
    y_score : array-like of shape (n_samples, n_labels)
        Target scores.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    coverage_error : float
        Coverage error.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)
    else:
        sample_weight = np.ones(y_true.shape[0])
    
    n_samples, n_labels = y_true.shape
    
    # Sort scores and get the rank of each score
    sorted_indices = np.argsort(y_score, axis=1)[:, ::-1]
    
    # Initialize coverage error
    coverage = 0.0
    
    for i in range(n_samples):
        true_labels = np.where(y_true[i])[0]
        if len(true_labels) == 0:
            continue
        
        max_rank = 0
        for label in true_labels:
            rank = np.where(sorted_indices[i] == label)[0][0]
            if rank > max_rank:
                max_rank = rank
        
        coverage += (max_rank + 1) * sample_weight[i]
    
    return coverage / np.sum(sample_weight)

