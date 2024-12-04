import numpy as np
from sklearn.utils import resample

def combo_moa(scores, n_buckets, method, bootstrap_estimators, random_state):
    n_samples, n_estimators = scores.shape
    combined_scores = np.zeros(n_samples)
    
    if method == 'static':
        bucket_size = n_estimators // n_buckets
        for i in range(n_buckets):
            start_idx = i * bucket_size
            end_idx = (i + 1) * bucket_size if i != n_buckets - 1 else n_estimators
            bucket_scores = scores[:, start_idx:end_idx]
            combined_scores += np.mean(bucket_scores, axis=1)
    elif method == 'dynamic':
        rng = np.random.default_rng(random_state)
        for i in range(n_buckets):
            if bootstrap_estimators:
                selected_estimators = resample(np.arange(n_estimators), replace=True, n_samples=n_estimators, random_state=rng)
            else:
                selected_estimators = rng.choice(np.arange(n_estimators), size=n_estimators, replace=False)
            bucket_scores = scores[:, selected_estimators]
            combined_scores += np.mean(bucket_scores, axis=1)
    else:
        raise ValueError("Method must be either 'static' or 'dynamic'")
    
    combined_scores /= n_buckets
    return combined_scores

def moa(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None):
    return combo_moa(scores, n_buckets, method, bootstrap_estimators, random_state)

