import numpy as np
from scipy.sparse import csr_matrix

def make_multilabel_classification(n_samples=100, n_features=20, n_classes=5, n_labels=2, length=50, allow_unlabeled=False, sparse=False, return_probabilities=False, random_state=None):
    """
    Generate a random multilabel classification problem.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.
    n_features : int, default=20
        The number of features.
    n_classes : int, default=5
        The number of classes.
    n_labels : int, default=2
        The average number of labels per instance.
    length : int, default=50
        The document length (number of words).
    allow_unlabeled : bool, default=False
        If True, some instances might not have any labels.
    sparse : bool, default=False
        If True, return X and Y as sparse matrices.
    return_probabilities : bool, default=False
        If True, return the prior class probability p_c and conditional probabilities p_w_c.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    X : ndarray or sparse matrix of shape (n_samples, n_features)
        The generated samples.
    Y : ndarray or sparse matrix of shape (n_samples, n_classes)
        The label sets.
    p_c : ndarray of shape (n_classes,), optional
        The prior class probability.
    p_w_c : ndarray of shape (n_classes, n_features), optional
        The conditional probabilities of features given classes.
    """
    rng = np.random.default_rng(random_state)

    # Generate prior class probabilities
    p_c = rng.dirichlet(alpha=np.ones(n_classes), size=1).flatten()

    # Generate conditional probabilities of features given classes
    p_w_c = rng.dirichlet(alpha=np.ones(n_features), size=n_classes)

    # Generate samples
    X = np.zeros((n_samples, n_features))
    Y = np.zeros((n_samples, n_classes))

    for i in range(n_samples):
        # Sample the number of labels for this instance
        n_labels_i = rng.poisson(lam=n_labels)
        if not allow_unlabeled:
            n_labels_i = max(1, n_labels_i)
        n_labels_i = min(n_labels_i, n_classes)

        # Sample the labels
        labels = rng.choice(n_classes, size=n_labels_i, replace=False, p=p_c)
        Y[i, labels] = 1

        # Sample the features
        for label in labels:
            words = rng.choice(n_features, size=length, replace=True, p=p_w_c[label])
            for word in words:
                X[i, word] += 1

    if sparse:
        X = csr_matrix(X)
        Y = csr_matrix(Y)

    if return_probabilities:
        return X, Y, p_c, p_w_c
    else:
        return X, Y

