def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    """
    Calculate the accuracy classification score.

    Parameters:
    y_true (list or array-like): Ground truth (correct) labels.
    y_pred (list or array-like): Predicted labels.
    normalize (bool): If True, return the fraction of correctly classified samples.
                      If False, return the number of correctly classified samples.
    sample_weight (list or array-like, optional): Sample weights.

    Returns:
    float or int: Accuracy score.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("The length of y_true and y_pred must be the same.")
    
    if sample_weight is None:
        sample_weight = [1] * len(y_true)
    elif len(sample_weight) != len(y_true):
        raise ValueError("The length of sample_weight must be the same as y_true and y_pred.")
    
    correct = sum(w for yt, yp, w in zip(y_true, y_pred, sample_weight) if yt == yp)
    total = sum(sample_weight)
    
    if normalize:
        return correct / total
    else:
        return correct

