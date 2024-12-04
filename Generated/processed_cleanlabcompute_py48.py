import numpy as np

def compute_py(ps, noise_matrix, inverse_noise_matrix, py_method, true_labels_class_counts):
    # Validate input shapes and parameters
    if ps is None or noise_matrix is None or inverse_noise_matrix is None or py_method is None or true_labels_class_counts is None:
        raise ValueError("All parameters must be provided.")
    
    if not isinstance(ps, np.ndarray) or not isinstance(noise_matrix, np.ndarray) or not isinstance(inverse_noise_matrix, np.ndarray) or not isinstance(true_labels_class_counts, np.ndarray):
        raise TypeError("ps, noise_matrix, inverse_noise_matrix, and true_labels_class_counts must be numpy arrays.")
    
    if ps.ndim != 1:
        raise ValueError("ps must be a 1-dimensional array.")
    
    if noise_matrix.shape != inverse_noise_matrix.shape:
        raise ValueError("noise_matrix and inverse_noise_matrix must have the same shape.")
    
    K = noise_matrix.shape[0]
    if noise_matrix.shape[1] != K:
        raise ValueError("noise_matrix and inverse_noise_matrix must be square matrices.")
    
    if true_labels_class_counts.shape != (K,):
        raise ValueError("true_labels_class_counts must be a 1-dimensional array of shape (K,).")
    
    if py_method not in ['cnt', 'eqn', 'marginal', 'marginal_ps']:
        raise ValueError("py_method must be one of 'cnt', 'eqn', 'marginal', or 'marginal_ps'.")
    
    # Compute py based on the specified method
    if py_method == 'cnt':
        py = true_labels_class_counts / np.sum(true_labels_class_counts)
    elif py_method == 'eqn':
        py = np.dot(inverse_noise_matrix, ps)
    elif py_method == 'marginal':
        py = np.dot(inverse_noise_matrix, np.dot(noise_matrix, true_labels_class_counts / np.sum(true_labels_class_counts)))
    elif py_method == 'marginal_ps':
        py = np.dot(inverse_noise_matrix, np.dot(noise_matrix, ps))
    
    # Ensure py is a 1-dimensional array
    py = np.squeeze(py)
    
    # Clip py to ensure values are between 0 and 1
    py = np.clip(py, 0, 1)
    
    return py

