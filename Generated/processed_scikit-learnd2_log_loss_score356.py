import numpy as np
from sklearn.metrics import log_loss
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples
import warnings
from sklearn.exceptions import UndefinedMetricWarning

def d2_log_loss_score(y_true, y_pred, sample_weight=None, labels=None):
    # Check if the number of samples is less than 2
    if _num_samples(y_pred) < 2:
        msg = "D^2 score is not well-defined with less than two samples."
        warnings.warn(msg, UndefinedMetricWarning)
        return float("nan")
    
    # Ensure y_true and y_pred are numpy arrays
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=None)
    
    # Check that y_true and y_pred have consistent lengths
    check_consistent_length(y_true, y_pred, sample_weight)
    
    # Determine the type of target
    y_type = type_of_target(y_true)
    
    if labels is None:
        labels = np.unique(y_true)
    
    # Calculate the log loss of the model
    model_log_loss = log_loss(y_true, y_pred, sample_weight=sample_weight, labels=labels)
    
    # Calculate the naive log loss
    if y_type == 'binary':
        # For binary classification
        p = np.average(y_true, weights=sample_weight)
        y_pred_naive = np.full_like(y_true, p)
        naive_log_loss = log_loss(y_true, y_pred_naive, sample_weight=sample_weight)
    else:
        # For multiclass classification
        class_counts = np.bincount(y_true, weights=sample_weight)
        class_probs = class_counts / np.sum(class_counts)
        y_pred_naive = np.tile(class_probs, (len(y_true), 1))
        naive_log_loss = log_loss(y_true, y_pred_naive, sample_weight=sample_weight, labels=labels)
    
    # Calculate the D^2 score
    d2_score = 1 - (model_log_loss / naive_log_loss)
    
    return d2_score

