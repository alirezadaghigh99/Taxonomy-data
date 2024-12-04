import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.utils import check_array, check_consistent_length

def _binary_uninterpolated_average_precision(y_true, y_score, sample_weight=None):
    # Sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        sample_weight = sample_weight[desc_score_indices]
    else:
        sample_weight = np.ones_like(y_true)

    # Calculate precision and recall
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true * sample_weight)[threshold_idxs]
    fps = np.cumsum((1 - y_true) * sample_weight)[threshold_idxs]

    precision = tps / (tps + fps)
    recall = tps / tps[-1]

    # Average precision calculation
    average_precision = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    return average_precision

def average_precision_score(y_true, y_score, average='macro', pos_label=1, sample_weight=None):
    y_true = check_array(y_true, ensure_2d=False)
    y_score = check_array(y_score, ensure_2d=False)
    check_consistent_length(y_true, y_score, sample_weight)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, 1)

    n_classes = y_score.shape[1]

    if n_classes == 1:
        return _binary_uninterpolated_average_precision(y_true.ravel(), y_score.ravel(), sample_weight)

    average_precision = np.zeros(n_classes)
    for i in range(n_classes):
        average_precision[i] = _binary_uninterpolated_average_precision(y_true[:, i], y_score[:, i], sample_weight)

    if average == 'micro':
        y_true = y_true.ravel()
        y_score = y_score.ravel()
        return _binary_uninterpolated_average_precision(y_true, y_score, sample_weight)
    elif average == 'samples':
        return np.mean([_binary_uninterpolated_average_precision(y_true[i], y_score[i], sample_weight) for i in range(y_true.shape[0])])
    elif average == 'weighted':
        weights = np.sum(y_true, axis=0)
        return np.average(average_precision, weights=weights)
    elif average == 'macro':
        return np.mean(average_precision)
    else:
        return average_precision

