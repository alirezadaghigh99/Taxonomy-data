import numpy as np

def make_circles(n_samples=100, shuffle=True, noise=None, random_state=None, factor=0.8):
    if isinstance(n_samples, int):
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
    elif isinstance(n_samples, tuple) and len(n_samples) == 2:
        n_samples_out, n_samples_in = n_samples
    else:
        raise ValueError("n_samples must be an integer or a tuple of two integers.")
    
    if not 0 < factor < 1:
        raise ValueError("factor must be between 0 and 1.")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate outer circle
    linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
    outer_circle = np.stack([np.cos(linspace_out), np.sin(linspace_out)], axis=1)
    
    # Generate inner circle
    linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
    inner_circle = factor * np.stack([np.cos(linspace_in), np.sin(linspace_in)], axis=1)
    
    # Combine the circles
    X = np.vstack([outer_circle, inner_circle])
    y = np.hstack([np.zeros(n_samples_out, dtype=int), np.ones(n_samples_in, dtype=int)])
    
    # Add noise
    if noise is not None:
        X += np.random.normal(scale=noise, size=X.shape)
    
    # Shuffle the dataset
    if shuffle:
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
    
    return X, y

