import numpy as np
from collections import Counter

def compute_sample_weight(class_weight, y, indices=None):
    """
    Estimate sample weights by class for unbalanced datasets.

    Parameters:
    - class_weight: dict, list of dicts, "balanced", or None
        Weights associated with classes.
    - y: array-like, sparse matrix
        Original class labels per sample.
    - indices: array-like of shape (n_subsample,), default=None
        Array of indices to be used in a subsample.

    Returns:
    - sample_weight_vect: numpy array of shape (n_samples,)
        Sample weights as applied to the original y.
    """
    if indices is not None:
        y = np.asarray(y)[indices]
    else:
        y = np.asarray(y)
    
    unique_classes = np.unique(y)
    class_counts = Counter(y)
    
    if class_weight == "balanced":
        total_samples = len(y)
        class_weight = {cls: total_samples / (len(unique_classes) * count) for cls, count in class_counts.items()}
    elif isinstance(class_weight, dict):
        pass
    elif class_weight is None:
        class_weight = {cls: 1.0 for cls in unique_classes}
    else:
        raise ValueError("class_weight must be 'balanced', a dict, or None")
    
    sample_weight_vect = np.array([class_weight[cls] for cls in y])
    
    return sample_weight_vect

