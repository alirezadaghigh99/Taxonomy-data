import numpy as np

def get_confident_thresholds(labels, predicted_probs):
    # Placeholder for the actual implementation of get_confident_thresholds
    # This function should return a dictionary with class-specific thresholds
    # For simplicity, let's assume it returns the mean predicted probability for each class
    thresholds = {}
    for i in range(predicted_probs.shape[1]):
        thresholds[i] = np.mean(predicted_probs[:, i])
    return thresholds

def _subtract_confident_thresholds(labels, predicted_probs, multi_label=False, thresholds=None):
    if thresholds is None:
        if labels is None:
            raise ValueError("Either labels or pre-calculated thresholds must be provided.")
        thresholds = get_confident_thresholds(labels, predicted_probs)
    
    # Convert thresholds to a numpy array for easier manipulation
    threshold_array = np.array([thresholds[i] for i in range(predicted_probs.shape[1])])
    
    # Subtract thresholds from predicted probabilities
    adjusted_probs = predicted_probs - threshold_array
    
    # Ensure no negative values by shifting
    adjusted_probs = np.maximum(adjusted_probs, 0)
    
    # Re-normalize probabilities
    if multi_label:
        # For multi-label, normalize each row independently
        row_sums = adjusted_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # To avoid division by zero
        adjusted_probs = adjusted_probs / row_sums
    else:
        # For single-label, normalize each row to sum to 1
        row_sums = adjusted_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # To avoid division by zero
        adjusted_probs = adjusted_probs / row_sums
    
    return adjusted_probs

