import numpy as np
from sklearn.utils import check_consistent_length, column_or_1d
from sklearn.utils.extmath import stable_cumsum

def det_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """
    Compute error rates for different probability thresholds in a binary classification task.

    Parameters:
    - y_true: ndarray of shape (n_samples), representing the true binary labels.
    - y_score: ndarray of shape (n_samples), representing target scores.
    - pos_label: int, float, bool, or str, default=None, indicating the label of the positive class.
    - sample_weight: array-like of shape (n_samples), default=None, representing sample weights.

    Returns:
    - fpr: ndarray of shape (n_thresholds), representing the false positive rate.
    - fnr: ndarray of shape (n_thresholds), representing the false negative rate.
    - thresholds: ndarray of shape (n_thresholds), representing decreasing score values.
    """
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    check_consistent_length(y_true, y_score, sample_weight)

    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight)

    if pos_label is None:
        pos_label = 1.0

    # Sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    sample_weight = sample_weight[desc_score_indices]

    # Determine the indices where the score changes
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # Accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * sample_weight)[threshold_idxs]
    fps = stable_cumsum((1 - y_true) * sample_weight)[threshold_idxs]

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    thresholds = y_score[threshold_idxs]

    # Total positive and negative weights
    P = tps[-1]
    N = fps[-1]

    # False positive rate
    fpr = fps / N
    # False negative rate
    fnr = (P - tps) / P

    return fpr, fnr, thresholds

