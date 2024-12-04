import numpy as np

def compute_centroid(coordinates):
    """
    Compute the centroid of a set of 3D coordinates.

    Parameters:
    coordinates (numpy.ndarray): A numpy array of shape (N, 3) representing the coordinates of atoms.

    Returns:
    numpy.ndarray: A numpy array of shape (3,) representing the centroid (x, y, z) of the provided coordinates.
    """
    if not isinstance(coordinates, np.ndarray) or coordinates.shape[1] != 3:
        raise ValueError("Input must be a numpy array of shape (N, 3).")
    
    # Calculate the centroid by averaging the coordinates along the first axis (rows)
    centroid = np.mean(coordinates, axis=0)
    
    return centroid

