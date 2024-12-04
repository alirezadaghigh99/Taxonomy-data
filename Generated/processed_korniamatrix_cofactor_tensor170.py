import numpy as np

def matrix_cofactor_tensor(matrix):
    """
    Computes the cofactor matrix for a given tensor of matrices with shape (*, 3, 3).
    
    Parameters:
    matrix (np.ndarray): A tensor of shape (*, 3, 3) representing the input matrices.
    
    Returns:
    np.ndarray: A tensor containing the cofactor matrices of the input matrices.
    
    Raises:
    Exception: If all input matrices are singular (i.e., their determinants are zero).
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Input tensor must have shape (*, 3, 3)")
    
    # Compute the determinant of each 3x3 matrix
    determinants = np.linalg.det(matrix)
    
    # Check if all matrices are singular
    if np.all(determinants == 0):
        raise Exception("All input matrices are singular (determinants are zero).")
    
    # Initialize the cofactor tensor with the same shape as the input tensor
    cofactor_tensor = np.empty_like(matrix)
    
    # Iterate over the last two dimensions (3x3 matrices)
    for i in range(3):
        for j in range(3):
            # Create the minor matrix by removing the i-th row and j-th column
            minor_matrix = np.delete(np.delete(matrix, i, axis=-2), j, axis=-1)
            
            # Compute the determinant of the minor matrix
            minor_determinants = np.linalg.det(minor_matrix)
            
            # Compute the cofactor for the (i, j) element
            cofactor_tensor[..., i, j] = ((-1) ** (i + j)) * minor_determinants
    
    # Transpose the cofactor matrix to get the correct cofactor matrix
    cofactor_tensor = np.transpose(cofactor_tensor, axes=(0, 2, 1))
    
    return cofactor_tensor

