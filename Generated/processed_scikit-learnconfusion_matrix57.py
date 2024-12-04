import numpy as np

def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    labels : array-like of shape (n_classes), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If None is given, those that appear at least once
        in y_true or y_pred are used in sorted order.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.

    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Confusion matrix whose i-th row and j-th
        column entry indicates the number of
        samples with true label being i-th class
        and predicted label being j-th class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    else:
        labels = np.asarray(labels)

    n_labels = labels.size
    label_to_index = {label: i for i, label in enumerate(labels)}

    cm = np.zeros((n_labels, n_labels), dtype=int)

    for i in range(len(y_true)):
        true_index = label_to_index[y_true[i]]
        pred_index = label_to_index[y_pred[i]]
        if sample_weight is None:
            cm[true_index, pred_index] += 1
        else:
            cm[true_index, pred_index] += sample_weight[i]

    if normalize is not None:
        with np.errstate(all='ignore'):
            if normalize == 'true':
                cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            elif normalize == 'pred':
                cm = cm.astype(float) / cm.sum(axis=0, keepdims=True)
            elif normalize == 'all':
                cm = cm.astype(float) / cm.sum()
            cm = np.nan_to_num(cm)

    return cm