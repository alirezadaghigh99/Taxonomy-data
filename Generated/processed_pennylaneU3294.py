import numpy as np

def U3(theta, phi, delta):
    """
    Returns the 2x2 unitary matrix for the given parameters theta, phi, and delta.
    
    Parameters:
    theta (float): The polar angle.
    phi (float): The azimuthal angle.
    delta (float): The quantum phase.
    
    Returns:
    numpy.ndarray: A 2x2 unitary matrix.
    """
    # Calculate the elements of the matrix
    cos_theta_2 = np.cos(theta / 2)
    sin_theta_2 = np.sin(theta / 2)
    exp_i_delta = np.exp(1j * delta)
    exp_i_phi = np.exp(1j * phi)
    exp_i_phi_delta = np.exp(1j * (phi + delta))
    
    # Construct the matrix
    matrix = np.array([
        [cos_theta_2, -exp_i_delta * sin_theta_2],
        [exp_i_phi * sin_theta_2, exp_i_phi_delta * cos_theta_2]
    ], dtype=complex)
    
    return matrix

