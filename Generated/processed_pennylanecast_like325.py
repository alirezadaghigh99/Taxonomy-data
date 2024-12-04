import numpy as np

def cast_like(tensor1, tensor2):
    """
    Cast tensor1 to the same data type as tensor2.

    Parameters:
    tensor1: numpy array, list, or tuple
    tensor2: numpy array, list, or tuple

    Returns:
    A new tensor1 cast to the same data type as tensor2.
    """
    # Convert tensor2 to a numpy array to determine its data type
    tensor2_np = np.array(tensor2)
    dtype = tensor2_np.dtype

    # Convert tensor1 to a numpy array and cast it to the same data type as tensor2
    tensor1_np = np.array(tensor1, dtype=dtype)

    # If tensor1 was originally a list or tuple, convert the numpy array back to the same type
    if isinstance(tensor1, list):
        return tensor1_np.tolist()
    elif isinstance(tensor1, tuple):
        return tuple(tensor1_np)
    else:
        return tensor1_np

