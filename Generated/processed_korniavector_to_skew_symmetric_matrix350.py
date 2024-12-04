import torch

def vector_to_skew_symmetric_matrix(v):
    """
    Converts a 3D vector or a batch of 3D vectors into their corresponding skew-symmetric matrices.
    
    Args:
        v (torch.Tensor): A tensor of shape (B, 3) or (3,) representing a batch of 3D vectors or a single 3D vector.
        
    Returns:
        torch.Tensor: A tensor of shape (B, 3, 3) or (3, 3) containing the skew-symmetric matrices.
        
    Raises:
        ValueError: If the input tensor does not have the correct shape.
    """
    if v.ndimension() == 1:
        if v.shape[0] != 3:
            raise ValueError("Input vector must have shape (3,)")
        v = v.unsqueeze(0)  # Convert to shape (1, 3) for consistent processing
    
    elif v.ndimension() == 2:
        if v.shape[1] != 3:
            raise ValueError("Each vector in the batch must have shape (3,)")
    
    else:
        raise ValueError("Input tensor must have shape (B, 3) or (3,)")
    
    B = v.shape[0]
    
    # Create the skew-symmetric matrix for each vector in the batch
    skew_matrices = torch.zeros((B, 3, 3), dtype=v.dtype, device=v.device)
    skew_matrices[:, 0, 1] = -v[:, 2]
    skew_matrices[:, 0, 2] = v[:, 1]
    skew_matrices[:, 1, 0] = v[:, 2]
    skew_matrices[:, 1, 2] = -v[:, 0]
    skew_matrices[:, 2, 0] = -v[:, 1]
    skew_matrices[:, 2, 1] = v[:, 0]
    
    if skew_matrices.shape[0] == 1:
        return skew_matrices.squeeze(0)  # Return shape (3, 3) for single vector input
    return skew_matrices

