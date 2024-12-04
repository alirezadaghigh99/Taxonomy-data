import numpy as np

def _estimate_joint_multilabel(labels, pred_probs):
    # Placeholder for the actual multi-label joint estimation logic
    # This function should return a (K, 2, 2) array
    K = pred_probs.shape[1]
    joint = np.zeros((K, 2, 2))
    for k in range(K):
        for i in range(2):
            for j in range(2):
                joint[k, i, j] = np.mean((labels[:, k] == i) & (pred_probs[:, k] >= 0.5) == j)
    return joint

def estimate_joint(labels, pred_probs, confident_joint=None, multi_label=False):
    labels = np.array(labels)
    pred_probs = np.array(pred_probs)
    
    if multi_label:
        if len(labels.shape) != 2 or len(pred_probs.shape) != 2:
            raise ValueError("For multi-label classification, labels and pred_probs should be 2D arrays.")
        K = pred_probs.shape[1]
    else:
        if len(labels.shape) != 1 or len(pred_probs.shape) != 2:
            raise ValueError("For multi-class classification, labels should be 1D and pred_probs should be 2D arrays.")
        K = pred_probs.shape[1]
    
    if confident_joint is None:
        if multi_label:
            confident_joint = _estimate_joint_multilabel(labels, pred_probs)
        else:
            confident_joint = np.zeros((K, K))
            for i in range(K):
                for j in range(K):
                    confident_joint[i, j] = np.mean((labels == i) & (np.argmax(pred_probs, axis=1) == j))
    
    confident_joint = np.array(confident_joint)
    
    if multi_label:
        if confident_joint.shape != (K, 2, 2):
            raise ValueError(f"Expected confident_joint shape to be ({K}, 2, 2) for multi-label classification.")
    else:
        if confident_joint.shape != (K, K):
            raise ValueError(f"Expected confident_joint shape to be ({K}, {K}) for multi-class classification.")
    
    # Normalize the confident_joint to ensure it sums to 1
    confident_joint_distribution = confident_joint / np.sum(confident_joint)
    
    return confident_joint_distribution

