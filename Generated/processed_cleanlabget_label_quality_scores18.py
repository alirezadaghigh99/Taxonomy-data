import numpy as np

def get_label_quality_scores(labels, predictions, method="outre"):
    """
    Calculate label quality scores for each example in a regression dataset.

    Parameters:
    labels (array-like): Raw labels from the original dataset.
    predictions (array-like): Predicted labels for each example.
    method (str): Scoring method to use (default is "outre").

    Returns:
    np.ndarray: Array of label quality scores, one score per example.
    """
    # Convert inputs to numpy arrays
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    
    # Check if inputs are valid
    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must have the same shape.")
    
    if method == "outre":
        # Calculate the absolute differences between labels and predictions
        differences = np.abs(labels - predictions)
        
        # Normalize the differences to get scores between 0 and 1
        max_diff = np.max(differences)
        if max_diff == 0:
            # If max_diff is 0, all labels and predictions are the same
            return np.ones_like(differences)
        
        scores = 1 - (differences / max_diff)
        return scores
    else:
        raise ValueError(f"Unknown method: {method}")

