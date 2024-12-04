import numpy as np

def _find_predicted_neq_given_multilabel(labels, pred_probs):
    """
    Helper function to handle multi-label classification.
    """
    # Convert labels to a numpy array if they are not already
    labels = np.array(labels)
    
    # Threshold to determine predicted labels from probabilities
    threshold = 0.5
    predicted_labels = (pred_probs >= threshold).astype(int)
    
    # Identify label issues where predicted labels do not match given labels
    label_issues = np.any(predicted_labels != labels, axis=1)
    
    return label_issues

def find_predicted_neq_given(labels, pred_probs, multi_label=False):
    """
    Identify label issues in the dataset.
    
    Parameters:
    - labels (np.ndarray or list): True labels of the dataset.
    - pred_probs (np.ndarray): Predicted probabilities from the model.
    - multi_label (bool, optional): Whether the task is multi-label classification.
    
    Returns:
    - np.ndarray: Boolean mask where True represents a label issue.
    """
    # Input validation
    if not isinstance(labels, (np.ndarray, list)):
        raise ValueError("labels should be a numpy array or a list.")
    if not isinstance(pred_probs, np.ndarray):
        raise ValueError("pred_probs should be a numpy array.")
    if not isinstance(multi_label, bool):
        raise ValueError("multi_label should be a boolean.")
    
    # Convert labels to a numpy array if they are not already
    labels = np.array(labels)
    
    if multi_label:
        return _find_predicted_neq_given_multilabel(labels, pred_probs)
    
    # For single-label classification
    # Get the predicted class with the highest probability
    predicted_labels = np.argmax(pred_probs, axis=1)
    
    # Identify label issues where predicted labels do not match given labels
    label_issues = predicted_labels != labels
    
    # High confidence threshold (e.g., 0.9)
    high_confidence = np.max(pred_probs, axis=1) >= 0.9
    
    # Combine label issues with high confidence
    label_issues = label_issues & high_confidence
    
    return label_issues

