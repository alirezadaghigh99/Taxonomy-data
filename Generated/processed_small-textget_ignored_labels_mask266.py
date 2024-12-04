import numpy as np
from scipy.sparse import csr_matrix

def get_ignored_labels_mask(y, ignored_label_value):
    if isinstance(y, csr_matrix):
        # Convert the csr_matrix to a dense array for easier manipulation
        dense_y = y.toarray()
        # Check if any value in each row is equal to ignored_label_value
        mask = np.any(dense_y == ignored_label_value, axis=1)
    else:
        # For non-csr_matrix, simply compare each element to ignored_label_value
        mask = (y == ignored_label_value)
    
    return mask

