import numpy as np
from scipy.sparse import csr_matrix

def empty_result(multi_label, num_classes, return_prediction=True, return_proba=True):
    if not return_prediction and not return_proba:
        raise ValueError("Invalid usage: At least one of 'prediction' or 'proba' must be True")
    
    result = {}
    
    if return_prediction:
        if multi_label:
            result['predictions'] = csr_matrix((0, num_classes), dtype=np.int64)
        else:
            result['predictions'] = np.empty((0,), dtype=np.int64)
    
    if return_proba:
        if multi_label:
            result['proba'] = csr_matrix((0, num_classes), dtype=float)
        else:
            result['proba'] = np.empty((0, num_classes), dtype=float)
    
    return result

