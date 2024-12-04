import numpy as np

def make_low_rank_matrix(n_samples, n_features, effective_rank, tail_strength=0.5, random_state=None):
    """
    Generate a mostly low-rank matrix with bell-shaped singular values.

    Parameters:
    - n_samples: int, number of samples (rows).
    - n_features: int, number of features (columns).
    - effective_rank: int, approximate rank of the matrix.
    - tail_strength: float, between 0 and 1, the relative importance of the tail singular values.
    - random_state: int or None, random seed for reproducibility.

    Returns:
    - X: ndarray of shape (n_samples, n_features), the generated low-rank matrix.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate the singular values with a bell-shaped profile
    singular_values = np.exp(-np.linspace(0, 4, min(n_samples, n_features))**2)
    
    # Adjust the singular values to have the desired effective rank
    singular_values = singular_values / singular_values.sum() * effective_rank
    
    # Add tail strength to the singular values
    tail = np.random.normal(size=singular_values.shape)
    singular_values = (1 - tail_strength) * singular_values + tail_strength * tail
    
    # Create random orthogonal matrices U and V
    U, _ = np.linalg.qr(np.random.randn(n_samples, n_samples))
    V, _ = np.linalg.qr(np.random.randn(n_features, n_features))
    
    # Construct the low-rank matrix
    S = np.zeros((n_samples, n_features))
    np.fill_diagonal(S, singular_values)
    
    X = U @ S @ V.T
    
    return X

