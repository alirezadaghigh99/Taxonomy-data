import numpy as np

def depth_from_point(R, t, X):
    """
    Return the depth of a point transformed by a rigid transform.

    Args:
       R: The rotation matrix with shape (*, 3, 3).
       t: The translation vector with shape (*, 3, 1).
       X: The 3d points with shape (*, 3).

    Returns:
       The depth value per point with shape (*, 1).
    """
    # Ensure inputs are numpy arrays
    R = np.asarray(R)
    t = np.asarray(t)
    X = np.asarray(X)
    
    # Check the shapes of the inputs
    assert R.shape[-2:] == (3, 3), "Rotation matrix R must have shape (*, 3, 3)"
    assert t.shape[-2:] == (3, 1), "Translation vector t must have shape (*, 3, 1)"
    assert X.shape[-1] == 3, "3D points X must have shape (*, 3)"
    
    # Reshape X to ensure it has the correct shape for matrix multiplication
    X = X[..., np.newaxis]  # Shape becomes (*, 3, 1)
    
    # Apply the rigid transformation
    X_transformed = np.matmul(R, X) + t  # Shape (*, 3, 1)
    
    # Extract the z-coordinate (depth) from the transformed points
    depth = X_transformed[..., 2, :]  # Shape (*, 1)
    
    return depth

