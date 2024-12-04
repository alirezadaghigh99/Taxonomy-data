import numpy as np

def cartesian(arrays, out=None):
    arrays = [np.asarray(a) for a in arrays]
    dtype = np.result_type(*arrays)
    
    n = np.prod([len(a) for a in arrays])
    if out is None:
        out = np.empty((n, len(arrays)), dtype=dtype)
    
    m = n // len(arrays[0])
    out[:, 0] = np.repeat(arrays[0], m)
    
    if arrays[1:]:
        cartesian(arrays[1:], out=out[:m, 1:])
        for j in range(1, len(arrays[0])):
            out[j*m:(j+1)*m, 1:] = out[:m, 1:]
    
    return out

