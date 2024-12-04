import numpy as np
from itertools import product

def _grid_from_X(X, percentiles, is_categorical, grid_resolution):
    X = np.asarray(X)
    n_samples, n_features = X.shape
    
    values = []
    
    for i in range(n_features):
        if is_categorical[i]:
            unique_values = np.unique(X[:, i])
            values.append(unique_values)
        else:
            lower, upper = np.percentile(X[:, i], percentiles)
            if grid_resolution > len(np.unique(X[:, i])):
                unique_values = np.unique(X[:, i])
                values.append(unique_values)
            else:
                values.append(np.linspace(lower, upper, grid_resolution))
    
    # Create the cartesian product of all values
    grid = np.array(list(product(*values)))
    
    return grid, values

