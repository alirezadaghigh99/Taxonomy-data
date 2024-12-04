import numpy as np
from sklearn.utils import check_array
from sklearn.utils import column_or_1d
from sklearn.utils import check_consistent_length

def calibration_curve(y_true, y_prob, pos_label=None, n_bins=5, strategy='uniform'):
    """
    Compute true and predicted probabilities for a calibration curve.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets.
    y_prob : array-like of shape (n_samples,)
        Probabilities of the positive class.
    pos_label : int, float, bool, or str, default=None
        The label of the positive class.
    n_bins : int, default=5
        Number of bins to discretize the [0, 1] interval.
    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.

    Returns
    -------
    prob_true : ndarray of shape (n_bins,) or smaller
        Proportion of samples whose class is the positive class in each bin.
    prob_pred : ndarray of shape (n_bins,) or smaller
        Mean predicted probability in each bin.
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)

    if pos_label is None:
        pos_label = 1.0

    y_true = (y_true == pos_label).astype(int)

    if strategy == 'uniform':
        bins = np.linspace(0., 1., n_bins + 1)
    elif strategy == 'quantile':
        bins = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    else:
        raise ValueError("Invalid value for 'strategy'. Choose 'uniform' or 'quantile'.")

    binids = np.digitize(y_prob, bins) - 1

    prob_true = np.zeros(n_bins)
    prob_pred = np.zeros(n_bins)

    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            prob_true[i] = np.mean(y_true[mask])
            prob_pred[i] = np.mean(y_prob[mask])

    return prob_true, prob_pred

