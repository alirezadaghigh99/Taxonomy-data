import numpy as np
from sklearn.utils import check_random_state, gen_batches, shuffle
from sklearn.linear_model import Lasso, lars_path
from sklearn.decomposition import DictionaryLearning
from sklearn.utils.extmath import randomized_svd

def dict_learning_online(X, n_components=2, alpha=1, max_iter=100, return_code=True,
                         dict_init=None, callback=None, batch_size=256, verbose=False,
                         shuffle=True, n_jobs=None, method='lars', random_state=None,
                         positive_dict=False, positive_code=False, method_max_iter=1000,
                         tol=1e-3, max_no_improvement=10):
    n_samples, n_features = X.shape
    random_state = check_random_state(random_state)
    
    if n_components is None:
        n_components = n_features

    if dict_init is None:
        U, S, V = randomized_svd(X, n_components)
        V = V * S[:, np.newaxis]
    else:
        V = dict_init

    if shuffle:
        X = shuffle(X, random_state=random_state)

    batches = gen_batches(n_samples, batch_size)
    n_batches = len(batches)

    if method == 'lars':
        def sparse_encode(X, dictionary):
            _, _, coefs = lars_path(dictionary.T, X.T, alpha=alpha, method='lasso')
            return coefs.T
    elif method == 'cd':
        def sparse_encode(X, dictionary):
            lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=method_max_iter, positive=positive_code)
            return np.array([lasso.fit(dictionary.T, x).coef_ for x in X])
    else:
        raise ValueError("Method must be 'lars' or 'cd'")

    best_cost = np.inf
    no_improvement = 0

    for iteration in range(max_iter):
        if verbose:
            print(f"Iteration {iteration + 1}/{max_iter}")

        for batch in batches:
            X_batch = X[batch]
            U_batch = sparse_encode(X_batch, V)
            if positive_dict:
                V = np.maximum(0, V + np.dot(U_batch.T, X_batch - np.dot(U_batch, V)))
            else:
                V += np.dot(U_batch.T, X_batch - np.dot(U_batch, V))
            V /= np.linalg.norm(V, axis=1)[:, np.newaxis]

        if callback is not None:
            callback(locals())

        cost = 0.5 * np.linalg.norm(X - np.dot(sparse_encode(X, V), V)) ** 2 + alpha * np.sum(np.abs(sparse_encode(X, V)))
        if verbose:
            print(f"Cost: {cost}")

        if cost < best_cost - tol:
            best_cost = cost
            no_improvement = 0
        else:
            no_improvement += 1

        if max_no_improvement is not None and no_improvement >= max_no_improvement:
            if verbose:
                print("Early stopping due to no improvement.")
            break

    U = sparse_encode(X, V)

    if return_code:
        return U, V, iteration + 1
    else:
        return V, iteration + 1