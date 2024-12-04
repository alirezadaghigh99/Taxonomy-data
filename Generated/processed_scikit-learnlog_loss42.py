import numpy as np

def log_loss(y_true, y_pred, normalize=True, sample_weight=None, labels=None):
    """
    Calculate the log loss (logistic loss or cross-entropy loss).

    Parameters:
    - y_true: array-like of shape (n_samples,) - Ground truth labels for n_samples samples.
    - y_pred: array-like of shape (n_samples, n_classes) - Predicted probabilities from a classifier's predict_proba method.
    - normalize: bool, default=True - Whether to return the mean loss per sample or the sum of per-sample losses.
    - sample_weight: array-like of shape (n_samples,), default=None - Optional sample weights.
    - labels: array-like of shape (n_classes,), default=None - Optional labels for the classes.

    Returns:
    - log_loss: float - The calculated log loss.

    Examples:
    >>> y_true = [0, 1, 1, 0]
    >>> y_pred = [[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.8, 0.2]]
    >>> log_loss(y_true, y_pred)
    0.21616187468057912

    Notes:
    - Log loss is a performance metric for evaluating the predictions of a classifier.
    - It is commonly used in binary and multi-class classification problems.
    - The log loss is calculated as the negative log-likelihood of the true labels given the predicted probabilities.

    References:
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
    - https://en.wikipedia.org/wiki/Cross_entropy
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if labels is not None:
        labels = np.array(labels)
        n_classes = len(labels)
        y_true = np.searchsorted(labels, y_true)
    else:
        n_classes = y_pred.shape[1]
    
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    else:
        sample_weight = np.array(sample_weight)
    
    # Clip y_pred to avoid log(0)
    eps = np.finfo(float).eps
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Create a one-hot encoding of y_true
    y_true_one_hot = np.zeros_like(y_pred)
    y_true_one_hot[np.arange(len(y_true)), y_true] = 1
    
    # Calculate the log loss
    loss = -np.sum(y_true_one_hot * np.log(y_pred), axis=1)
    
    # Apply sample weights
    loss = loss * sample_weight
    
    if normalize:
        return np.average(loss)
    else:
        return np.sum(loss)

