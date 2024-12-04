import numpy as np
from sklearn.utils import check_random_state

def aom(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None):
    """
    Implements the Average of Maximum (AOM) ensemble method for combining multiple estimators.

    Parameters:
    - scores: numpy array of shape (n_samples, n_estimators) representing the score matrix outputted from various estimators
    - n_buckets: integer specifying the number of subgroups to build (default value is 5)
    - method: string specifying the method for building subgroups ('static' or 'dynamic', default value is 'static')
    - bootstrap_estimators: boolean indicating whether estimators are drawn with replacement (default value is False)
    - random_state: integer, RandomState instance, or None specifying the seed for the random number generator (default value is None)

    Returns:
    - combined_scores: numpy array of shape (n_samples,) representing the combined outlier scores calculated using the Average of Maximum method
    """
    n_samples, n_estimators = scores.shape
    rng = check_random_state(random_state)
    
    if method not in ['static', 'dynamic']:
        raise ValueError("Method must be either 'static' or 'dynamic'")
    
    if n_buckets > n_estimators:
        raise ValueError("Number of buckets cannot be greater than the number of estimators")
    
    combined_scores = np.zeros(n_samples)
    
    if method == 'static':
        # Static method: divide estimators into n_buckets subgroups
        indices = np.arange(n_estimators)
        if bootstrap_estimators:
            indices = rng.choice(indices, size=n_estimators, replace=True)
        bucket_size = n_estimators // n_buckets
        for i in range(n_buckets):
            start_idx = i * bucket_size
            end_idx = (i + 1) * bucket_size if i < n_buckets - 1 else n_estimators
            bucket_scores = scores[:, indices[start_idx:end_idx]]
            combined_scores += np.max(bucket_scores, axis=1)
    else:
        # Dynamic method: randomly assign estimators to n_buckets subgroups
        for i in range(n_buckets):
            if bootstrap_estimators:
                bucket_indices = rng.choice(n_estimators, size=n_estimators // n_buckets, replace=True)
            else:
                bucket_indices = rng.choice(n_estimators, size=n_estimators // n_buckets, replace=False)
            bucket_scores = scores[:, bucket_indices]
            combined_scores += np.max(bucket_scores, axis=1)
    
    combined_scores /= n_buckets
    return combined_scores