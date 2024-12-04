import numpy as np

def U2(phi, delta):
    """
    Returns the matrix representation of the U2 gate.
    
    Parameters:
    phi (float): The azimuthal angle.
    delta (float): The quantum phase.
    
    Returns:
    numpy.ndarray: The 2x2 matrix representation of the U2 gate.
    """
    # Calculate the complex exponentials
    exp_i_delta = np.exp(1j * delta)
    exp_i_phi = np.exp(1j * phi)
    exp_i_phi_plus_delta = np.exp(1j * (phi + delta))
    
    # Construct the U2 matrix
    U2_matrix = (1 / np.sqrt(2)) * np.array([
        [1, -exp_i_delta],
        [exp_i_phi, exp_i_phi_plus_delta]
    ])
    
    return U2_matrix

