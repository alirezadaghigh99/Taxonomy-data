import numpy as np

def top_k_accuracy_score(y_true, y_score, k=1, normalize=True, sample_weight=None, labels=None):
    """
    Calculate the top-k accuracy classification score.

    Parameters:
    - y_true: array-like of shape (n_samples,) representing the true labels.
    - y_score: array-like of shape (n_samples,) or (n_samples, n_classes) representing the target scores.
    - k: int, number of most likely outcomes considered to find the correct label.
    - normalize: bool, whether to return the fraction of correctly classified samples or the number of correctly classified samples.
    - sample_weight: array-like of shape (n_samples,) representing sample weights.
    - labels: array-like of shape (n_classes,) representing the list of labels that index the classes in y_score.

    Returns:
    - float, top-k accuracy score.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    if labels is not None:
        labels = np.asarray(labels)
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        y_true = np.array([label_to_index[label] for label in y_true])
    
    if y_score.ndim == 1:
        y_score = y_score[:, np.newaxis]
    
    n_samples = y_true.shape[0]
    n_classes = y_score.shape[1]
    
    if sample_weight is None:
        sample_weight = np.ones(n_samples)
    else:
        sample_weight = np.asarray(sample_weight)
    
    # Get the indices of the top-k scores
    top_k_indices = np.argsort(y_score, axis=1)[:, -k:]
    
    # Check if the true label is in the top-k indices
    correct = np.any(top_k_indices == y_true[:, np.newaxis], axis=1)
    
    # Calculate the weighted sum of correct predictions
    correct_sum = np.sum(correct * sample_weight)
    
    if normalize:
        return correct_sum / np.sum(sample_weight)
    else:
        return correct_sum

