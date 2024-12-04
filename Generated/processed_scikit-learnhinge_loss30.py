import numpy as np

def hinge_loss(y_true, pred_decision, labels=None, sample_weight=None):
    """
    Calculate the average hinge loss for binary or multiclass classification tasks.

    Parameters:
    - y_true: array-like, shape (n_samples,)
        True target values, encoded as integers (+1 and -1 for binary classification).
    - pred_decision: array-like, shape (n_samples,) or (n_samples, n_classes)
        Predicted decision values.
    - labels: array-like, shape (n_classes,), optional
        All the labels for multiclass hinge loss.
    - sample_weight: array-like, shape (n_samples,), optional
        Sample weights.

    Returns:
    - float
        The average hinge loss.
    """
    y_true = np.asarray(y_true)
    pred_decision = np.asarray(pred_decision)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if sample_weight.shape[0] != y_true.shape[0]:
            raise ValueError("sample_weight and y_true must have the same length.")
    
    if labels is not None:
        labels = np.asarray(labels)
    
    n_samples = y_true.shape[0]
    
    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1-dimensional array.")
    
    if pred_decision.ndim == 1:
        # Binary classification
        if not np.all(np.isin(y_true, [-1, 1])):
            raise ValueError("For binary classification, y_true must contain only -1 and 1.")
        
        margin = y_true * pred_decision
        losses = np.maximum(0, 1 - margin)
    else:
        # Multiclass classification
        if labels is None:
            labels = np.unique(y_true)
        
        n_classes = len(labels)
        if pred_decision.shape[1] != n_classes:
            raise ValueError("pred_decision shape does not match the number of labels.")
        
        correct_class_scores = pred_decision[np.arange(n_samples), y_true]
        margins = pred_decision - correct_class_scores[:, np.newaxis] + 1
        margins[np.arange(n_samples), y_true] = 0
        losses = np.maximum(0, margins)
        losses = np.sum(losses, axis=1)
    
    if sample_weight is not None:
        average_loss = np.average(losses, weights=sample_weight)
    else:
        average_loss = np.mean(losses)
    
    return average_loss

