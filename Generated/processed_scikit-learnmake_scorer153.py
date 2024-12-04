from sklearn.base import is_classifier
from sklearn.utils.validation import check_is_fitted
import numpy as np

def make_scorer(score_func, response_method='auto', greater_is_better=True, needs_proba=False, needs_threshold=False, **kwargs):
    """
    Create a scorer from a performance metric or loss function.

    Parameters:
    - score_func: callable
        The performance metric or loss function to use for scoring.
    - response_method: str, default='auto'
        Specifies the method to use for prediction. Options are 'predict', 'predict_proba', 'decision_function', or 'auto'.
    - greater_is_better: bool, default=True
        Whether a higher score indicates better performance.
    - needs_proba: bool, default=False
        Whether the score_func requires probability estimates.
    - needs_threshold: bool, default=False
        Whether the score_func requires decision function or predict_proba output.
    - **kwargs: additional keyword arguments
        Additional parameters to pass to the score_func.

    Returns:
    - scorer: callable
        A callable object that computes a scalar score.
    """
    def scorer(estimator, X, y_true):
        check_is_fitted(estimator)
        
        if response_method == 'auto':
            if needs_proba:
                response_method_ = 'predict_proba'
            elif needs_threshold:
                response_method_ = 'decision_function'
            else:
                response_method_ = 'predict'
        else:
            response_method_ = response_method

        if response_method_ == 'predict_proba':
            y_pred = estimator.predict_proba(X)
        elif response_method_ == 'decision_function':
            y_pred = estimator.decision_function(X)
        else:
            y_pred = estimator.predict(X)

        score = score_func(y_true, y_pred, **kwargs)
        
        if not greater_is_better:
            score = -score
        
        return score

    return scorer

