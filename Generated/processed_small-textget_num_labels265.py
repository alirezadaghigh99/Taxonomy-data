import numpy as np
from scipy.sparse import csr_matrix

def get_num_labels(y):
    # Check if the shape of y is 0
    if np.shape(y) == (0,):
        raise ValueError('Invalid labeling: Cannot contain 0 labels')
    
    # Check if y is an instance of csr_matrix
    if isinstance(y, csr_matrix):
        return y.indices.max() + 1
    
    # Otherwise, return the maximum value of y plus 1
    return np.max(y) + 1

