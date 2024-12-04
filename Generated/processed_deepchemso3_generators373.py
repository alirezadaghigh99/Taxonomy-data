import torch

def so3_generators(k):
    """
    Generate the generators of the special orthogonal group SO(3).

    Parameters
    ----------
    k : int
        The representation index, which determines the order of the representation.

    Returns
    -------
    torch.Tensor
        A stack of three SO(3) generators, corresponding to J_x, J_y, and J_z.
    """
    if k < 1:
        raise ValueError("The representation index k must be a positive integer.")

    # Define the dimension of the representation
    dim = 2 * k + 1

    # Initialize the generators
    J_x = torch.zeros((dim, dim), dtype=torch.float64)
    J_y = torch.zeros((dim, dim), dtype=torch.float64)
    J_z = torch.zeros((dim, dim), dtype=torch.float64)

    # Fill the generators
    for m in range(-k, k):
        idx = m + k
        # J_x
        if idx + 1 < dim:
            J_x[idx, idx + 1] = 0.5 * torch.sqrt((k - m) * (k + m + 1))
            J_x[idx + 1, idx] = 0.5 * torch.sqrt((k + m + 1) * (k - m))
        
        # J_y
        if idx + 1 < dim:
            J_y[idx, idx + 1] = -0.5j * torch.sqrt((k - m) * (k + m + 1))
            J_y[idx + 1, idx] = 0.5j * torch.sqrt((k + m + 1) * (k - m))
        
        # J_z
        J_z[idx, idx] = m

    # Stack the generators into a single tensor
    generators = torch.stack((J_x, J_y, J_z))

    return generators

