import numpy as np

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    Converts a 3x3 rotation matrix to a 4D quaternion vector.
    
    Parameters:
    rotation_matrix (np.ndarray): A tensor of shape (*, 3, 3) representing the rotation matrix.
    eps (float): A small value to avoid zero division.
    
    Returns:
    np.ndarray: A tensor of shape (*, 4) representing the quaternion in (w, x, y, z) format.
    """
    if not isinstance(rotation_matrix, np.ndarray):
        raise TypeError("Input rotation_matrix must be a numpy ndarray.")
    
    if rotation_matrix.ndim < 2 or rotation_matrix.shape[-2:] != (3, 3):
        raise ValueError("Input rotation_matrix must have shape (*, 3, 3).")
    
    # Initialize the output quaternion tensor
    quaternions = np.zeros(rotation_matrix.shape[:-2] + (4,))
    
    # Extract the elements of the rotation matrix
    R = rotation_matrix
    trace = np.trace(R, axis1=-2, axis2=-1)
    
    # Compute the quaternion components
    w = np.sqrt(1.0 + trace + eps) / 2.0
    x = np.sqrt(1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2] + eps) / 2.0
    y = np.sqrt(1.0 + R[..., 1, 1] - R[..., 0, 0] - R[..., 2, 2] + eps) / 2.0
    z = np.sqrt(1.0 + R[..., 2, 2] - R[..., 0, 0] - R[..., 1, 1] + eps) / 2.0
    
    # Determine the signs of the quaternion components
    x = np.copysign(x, R[..., 2, 1] - R[..., 1, 2])
    y = np.copysign(y, R[..., 0, 2] - R[..., 2, 0])
    z = np.copysign(z, R[..., 1, 0] - R[..., 0, 1])
    
    # Assign the components to the output tensor
    quaternions[..., 0] = w
    quaternions[..., 1] = x
    quaternions[..., 2] = y
    quaternions[..., 3] = z
    
    return quaternions

