from sklearn.metrics import get_scorer as sklearn_get_scorer
from copy import deepcopy

def get_scorer(scoring):
    """
    Retrieve a scorer based on the input scoring method.

    Parameters:
    scoring (str, callable, or None): The scoring method to retrieve. This can be:
        - A string representing the name of a scorer (e.g., 'accuracy', 'f1').
        - A callable that takes (estimator, X, y) and returns a score.
        - None, in which case the function returns None.

    Returns:
    callable or None: The scorer object based on the input scoring method.

    Raises:
    ValueError: If the input scoring value is not a valid string, callable, or None.

    Examples:
    >>> get_scorer('accuracy')
    <function accuracy_scorer at 0x...>

    >>> def custom_scorer(estimator, X, y):
    ...     return estimator.score(X, y)
    >>> get_scorer(custom_scorer)
    <function custom_scorer at 0x...>

    >>> get_scorer(None)
    None

    >>> get_scorer('invalid_scorer')
    ValueError: Invalid scoring method: 'invalid_scorer'
    """
    if isinstance(scoring, str):
        try:
            return deepcopy(sklearn_get_scorer(scoring))
        except ValueError:
            raise ValueError(f"Invalid scoring method: '{scoring}'")
    elif callable(scoring):
        return scoring
    elif scoring is None:
        return None
    else:
        raise ValueError(f"Invalid scoring method: '{scoring}'")

