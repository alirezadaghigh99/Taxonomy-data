import numpy as np
from sklearn.utils import check_random_state

def _smacof_single(dissimilarities, metric=True, n_components=2, init=None, max_iter=300, verbose=0, eps=1e-3, random_state=None, normalized_stress=False):
    """
    Computes multidimensional scaling using SMACOF algorithm.

    Parameters
    ----------
    dissimilarities : ndarray of shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Must be symmetric.

    metric : bool, default=True
        Compute metric or nonmetric SMACOF algorithm.
        When ``False`` (i.e. non-metric MDS), dissimilarities with 0 are considered as
        missing values.

    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.

    init : ndarray of shape (n_samples, n_components), default=None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.

    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, default=0
        Level of verbosity.

    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare
        convergence. The value of `eps` should be tuned separately depending
        on whether or not `normalized_stress` is being used.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    normalized_stress : bool, default=False
        Whether use and return normed stress value (Stress-1) instead of raw
        stress calculated by default. Only supported in non-metric MDS. The
        caller must ensure that if `normalized_stress=True` then `metric=False`

        .. versionadded:: 1.2

    Returns
    -------
    X : ndarray of shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.

    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).
        If `normalized_stress=True`, and `metric=False` returns Stress-1.
        A value of 0 indicates "perfect" fit, 0.025 excellent, 0.05 good,
        0.1 fair, and 0.2 poor [1]_.

    n_iter : int
        The number of iterations corresponding to the best stress.
    """
    n_samples = dissimilarities.shape[0]
    random_state = check_random_state(random_state)

    if init is None:
        X = random_state.rand(n_samples, n_components)
    else:
        X = init

    old_stress = None

    for it in range(max_iter):
        # Compute distance matrix
        dist = np.sqrt(((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=2))

        if metric:
            disparities = dissimilarities
        else:
            disparities = np.copy(dissimilarities)
            disparities[disparities == 0] = np.nan
            disparities = np.nan_to_num(disparities)

        # Compute stress
        stress = ((disparities - dist) ** 2).sum() / 2

        if normalized_stress and not metric:
            stress /= (disparities ** 2).sum() / 2

        if verbose:
            print(f"Iteration {it + 1}, stress: {stress}")

        if old_stress is not None and abs(old_stress - stress) < eps:
            break

        old_stress = stress

        # Update X
        B = -disparities / dist
        B[np.arange(n_samples), np.arange(n_samples)] = 0
        B[np.arange(n_samples), np.arange(n_samples)] = -B.sum(axis=1)
        X = np.dot(B, X) / n_samples

    return X, stress, it + 1