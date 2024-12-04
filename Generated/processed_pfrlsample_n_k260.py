import numpy as np

def sample_n_k(n, k):
    if k > n or k < 0:
        raise ValueError("k must be non-negative and less than or equal to n.")
    
    if k == 0:
        return np.array([], dtype=int)
    
    if 3 * k >= n:
        # Use numpy's random.choice to sample k elements without replacement
        return np.random.choice(n, k, replace=False)
    else:
        # Sample 2k elements and ensure they are distinct
        sampled_set = set()
        while len(sampled_set) < k:
            # Sample 2k elements
            samples = np.random.choice(n, 2 * k, replace=True)
            # Add them to the set to ensure uniqueness
            sampled_set.update(samples)
        
        # Convert the set to a list and return the first k elements
        return np.array(list(sampled_set)[:k])

