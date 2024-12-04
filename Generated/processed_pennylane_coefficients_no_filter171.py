import numpy as np

def _coefficients_no_filter(f, degree, use_broadcasting=True):
    """
    Compute the first 2d+1 Fourier coefficients for a 2π-periodic function.

    Parameters:
    f (callable): The 2π-periodic function to analyze.
    degree (int or tuple): The maximum frequency. If tuple, it should be (d1, d2, ..., dn).
    use_broadcasting (bool): Whether to use broadcasting for simultaneous function evaluations.

    Returns:
    np.ndarray: Array of complex numbers containing the Fourier coefficients.
    """
    if isinstance(degree, int):
        degrees = (degree,)
    else:
        degrees = degree

    # Determine the number of dimensions
    num_dims = len(degrees)
    
    # Create a meshgrid for the evaluation points
    points_per_dim = [2*d + 1 for d in degrees]
    total_points = np.prod(points_per_dim)
    
    # Create the evaluation points
    grids = [np.linspace(0, 2*np.pi, num=p, endpoint=False) for p in points_per_dim]
    mesh = np.meshgrid(*grids, indexing='ij')
    eval_points = np.stack(mesh, axis=-1).reshape(-1, num_dims)
    
    # Evaluate the function at the grid points
    if use_broadcasting:
        values = f(*eval_points.T)
    else:
        values = np.array([f(*point) for point in eval_points])
    
    # Reshape values to match the grid shape
    values = values.reshape(points_per_dim)
    
    # Compute the Fourier coefficients using the FFT
    coeffs = np.fft.fftn(values) / total_points
    
    # Shift the zero frequency component to the center
    coeffs = np.fft.fftshift(coeffs)
    
    return coeffs

