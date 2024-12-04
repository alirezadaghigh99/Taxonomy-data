import numpy as np

def dcg_score(y_true, y_score, k=None, ignore_ties=False):
    """Compute Discounted Cumulative Gain."""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    
    if ignore_ties:
        gains = y_true
    else:
        gains = 2 ** y_true - 1
    
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=None, sample_weight=None, ignore_ties=False):
    """Compute Normalized Discounted Cumulative Gain."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    if sample_weight is None:
        sample_weight = np.ones(y_true.shape[0])
    else:
        sample_weight = np.asarray(sample_weight)
    
    if k is None:
        k = y_true.shape[1]
    
    scores = []
    for i in range(y_true.shape[0]):
        actual_dcg = dcg_score(y_true[i], y_score[i], k, ignore_ties)
        ideal_dcg = dcg_score(y_true[i], y_true[i], k, ignore_ties)
        if ideal_dcg == 0:
            scores.append(0.0)
        else:
            scores.append(actual_dcg / ideal_dcg)
    
    return np.average(scores, weights=sample_weight)

