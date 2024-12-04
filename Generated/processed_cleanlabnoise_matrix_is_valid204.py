import numpy as np

def noise_matrix_is_valid(noise_matrix: np.ndarray, py: np.ndarray, verbose: bool = False) -> bool:
    """
    Check if the given noise_matrix is a learnable matrix.

    Parameters:
    noise_matrix (np.ndarray): A square matrix where element (i, j) represents the probability of class i being 
                               mislabeled as class j.
    py (np.ndarray): A vector where element i represents the prior probability of class i.
    verbose (bool): If True, print detailed information about the validation process.

    Returns:
    bool: True if the noise matrix is learnable, False otherwise.
    """
    # Check if noise_matrix is square
    if noise_matrix.shape[0] != noise_matrix.shape[1]:
        if verbose:
            print("Noise matrix is not square.")
        return False

    # Check if py is a valid probability distribution
    if not np.isclose(np.sum(py), 1):
        if verbose:
            print("py does not sum to 1.")
        return False

    if np.any(py < 0) or np.any(py > 1):
        if verbose:
            print("py contains invalid probabilities.")
        return False

    # Calculate the effective noise rate for each class
    effective_noise_rates = np.sum(noise_matrix * py[:, np.newaxis], axis=0)

    # Check if the effective noise rate for any class is greater than or equal to 1
    if np.any(effective_noise_rates >= 1):
        if verbose:
            print("Effective noise rate for one or more classes is >= 1.")
        return False

    # Check if the noise matrix is stochastic (rows sum to 1)
    if not np.allclose(np.sum(noise_matrix, axis=1), 1):
        if verbose:
            print("Rows of the noise matrix do not sum to 1.")
        return False

    # Check if the noise matrix is non-negative
    if np.any(noise_matrix < 0):
        if verbose:
            print("Noise matrix contains negative probabilities.")
        return False

    if verbose:
        print("Noise matrix is valid and learnable.")

    return True

