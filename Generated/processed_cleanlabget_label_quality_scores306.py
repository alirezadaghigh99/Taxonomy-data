import numpy as np
from scipy.stats import entropy

def get_label_quality_scores(labels, pred_probs, method="self_confidence", adjust_pred_probs=False):
    """
    Compute label quality scores for a multi-class classification dataset.

    Parameters
    ----------
    labels : np.ndarray
        A discrete vector of noisy labels, i.e. some labels may be erroneous.
        Format: labels = np.ndarray([1,0,2,1,1,0...])
    
    pred_probs : np.ndarray
        An array of shape (N, K) of model-predicted probabilities, P(label=k|x).
    
    method : {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default="self_confidence"
        Label quality scoring method.
    
    adjust_pred_probs : bool, optional
        Account for class imbalance in the label-quality scoring by adjusting predicted probabilities.
    
    Returns
    -------
    label_quality_scores : np.ndarray
        Contains one score (between 0 and 1) per example.
        Lower scores indicate more likely mislabeled examples.
    """
    
    def adjust_probabilities(pred_probs):
        class_counts = np.bincount(labels)
        class_probs = class_counts / len(labels)
        adjusted_probs = pred_probs - class_probs
        adjusted_probs = np.maximum(adjusted_probs, 0)
        adjusted_probs /= adjusted_probs.sum(axis=1, keepdims=True)
        return adjusted_probs
    
    if adjust_pred_probs:
        pred_probs = adjust_probabilities(pred_probs)
    
    scores = np.zeros(len(labels))
    
    for i, (label, probs) in enumerate(zip(labels, pred_probs)):
        if method == "self_confidence":
            score = probs[label]
        elif method == "normalized_margin":
            max_other_prob = np.max(np.delete(probs, label))
            score = probs[label] - max_other_prob
        elif method == "confidence_weighted_entropy":
            score = entropy(probs) / probs[label]
        else:
            raise ValueError("Invalid method. Choose from 'self_confidence', 'normalized_margin', 'confidence_weighted_entropy'.")
        
        # Normalize scores to be between 0 and 1
        if method == "normalized_margin":
            score = (score + 1) / 2  # Normalizing to [0, 1]
        elif method == "confidence_weighted_entropy":
            score = 1 - score  # Inverting to make lower scores indicate more likely mislabeled
        
        scores[i] = score
    
    return scores

