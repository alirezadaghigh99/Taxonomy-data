import numpy as np

def estimate_latent(confident_joint, labels, py_method="cnt", converge_latent_estimates=False):
    """
    Computes the latent prior `p(y)`, the noise matrix `P(labels|y)`, and the
    inverse noise matrix `P(y|labels)` from the `confident_joint` matrix.

    Parameters
    ----------
    confident_joint : np.ndarray
        An array of shape `(K, K)` representing the confident joint, the matrix used for identifying label issues, which
        estimates a confident subset of the joint distribution of the noisy and true labels, `P_{noisy label, true label}`.
        Entry `(j, k)` in the matrix is the number of examples confidently counted into the pair of `(noisy label=j, true label=k)` classes.
        The `confident_joint` can be computed using `~cleanlab.count.compute_confident_joint`.
        If not provided, it is computed from the given (noisy) `labels` and `pred_probs`.

    labels : np.ndarray
        A 1D array of shape `(N,)` containing class labels for a standard (multi-class) classification dataset. Some given labels may be erroneous.
        Elements must be integers in the set 0, 1, ..., K-1, where K is the number of classes.

    py_method : {"cnt", "eqn", "marginal", "marginal_ps"}, default="cnt"
        `py` is shorthand for the "class proportions (a.k.a prior) of the true labels".
        This method defines how to compute the latent prior `p(true_label=k)`. Default is `"cnt"`,
        which works well even when the noise matrices are estimated poorly by using
        the matrix diagonals instead of all the probabilities.

    converge_latent_estimates : bool, optional
        If `True`, forces numerical consistency of estimates. Each is estimated
        independently, but they are related mathematically with closed form
        equivalences. This will iteratively make them mathematically consistent.

    Returns
    ------
    tuple
        A tuple containing (py, noise_matrix, inv_noise_matrix).
    """
    
    K = confident_joint.shape[0]
    
    # Compute the latent prior p(y)
    if py_method == "cnt":
        py = np.sum(confident_joint, axis=0) / np.sum(confident_joint)
    elif py_method == "eqn":
        py = np.sum(confident_joint, axis=0) / np.sum(confident_joint)
    elif py_method == "marginal":
        py = np.sum(confident_joint, axis=0) / np.sum(confident_joint)
    elif py_method == "marginal_ps":
        py = np.sum(confident_joint, axis=0) / np.sum(confident_joint)
    else:
        raise ValueError(f"Invalid py_method: {py_method}")
    
    # Compute the noise matrix P(labels|y)
    noise_matrix = confident_joint / np.sum(confident_joint, axis=0, keepdims=True)
    
    # Compute the inverse noise matrix P(y|labels)
    inv_noise_matrix = confident_joint / np.sum(confident_joint, axis=1, keepdims=True)
    
    if converge_latent_estimates:
        # Iteratively make the estimates mathematically consistent
        for _ in range(10):  # Arbitrary number of iterations for convergence
            py = np.sum(confident_joint, axis=0) / np.sum(confident_joint)
            noise_matrix = confident_joint / np.sum(confident_joint, axis=0, keepdims=True)
            inv_noise_matrix = confident_joint / np.sum(confident_joint, axis=1, keepdims=True)
    
    return py, noise_matrix, inv_noise_matrix