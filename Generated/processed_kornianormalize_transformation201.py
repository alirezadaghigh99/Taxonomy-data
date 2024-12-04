import numpy as np

def normalize_transformation(M, eps=1e-8):
    # Ensure M has at least 2 dimensions
    assert M.ndim >= 2, "Input matrix M must have at least 2 dimensions"
    
    # Get the shape of the input matrix
    shape = M.shape
    
    # Extract the value at the last row and column
    last_value = M[-1, -1]
    
    # Avoid division by zero by adding a small epsilon value
    if abs(last_value) < eps:
        last_value = eps
    
    # Normalize the matrix
    normalized_M = M / last_value
    
    # Set the last element to exactly 1
    normalized_M[-1, -1] = 1.0
    
    return normalized_M

