import numpy as np

def _compute_label_quality_scores(labels, predictions, method="objectlab", aggregation_weights=None, threshold=None, overlapping_label_check=True, verbose=True):
    if not isinstance(labels, list) or not all(isinstance(label, dict) for label in labels):
        raise ValueError("Labels must be a list of dictionaries.")
    
    if not isinstance(predictions, list) or not all(isinstance(pred, np.ndarray) for pred in predictions):
        raise ValueError("Predictions must be a list of numpy arrays.")
    
    if method not in ["objectlab"]:
        raise ValueError(f"Unsupported method: {method}")
    
    if aggregation_weights is not None and not isinstance(aggregation_weights, dict):
        raise ValueError("Aggregation weights must be a dictionary with string keys and float values.")
    
    if threshold is not None and not isinstance(threshold, (int, float)):
        raise ValueError("Threshold must be a float or int.")
    
    if not isinstance(overlapping_label_check, bool):
        raise ValueError("Overlapping label check must be a boolean.")
    
    if not isinstance(verbose, bool):
        raise ValueError("Verbose must be a boolean.")
    
    # Prune extra bounding boxes if necessary
    pruned_labels = []
    for label in labels:
        if overlapping_label_check:
            # Implement logic to prune overlapping bounding boxes
            # This is a placeholder for actual pruning logic
            pruned_label = {k: v for k, v in label.items() if not isinstance(v, list) or len(v) <= 1}
        else:
            pruned_label = label
        pruned_labels.append(pruned_label)
    
    # Compute label quality scores based on the specified method
    scores = []
    if method == "objectlab":
        for label, prediction in zip(pruned_labels, predictions):
            # Implement the specific scoring logic for "objectlab"
            # This is a placeholder for actual scoring logic
            score = np.mean(prediction)  