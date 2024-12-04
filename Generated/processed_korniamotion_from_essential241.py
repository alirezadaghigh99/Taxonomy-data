import torch

def motion_from_essential(E_mat):
    """
    Decomposes an essential matrix into the four possible poses.
    
    Args:
    - E_mat (torch.Tensor): Essential matrix of shape (*, 3, 3)
    
    Returns:
    - Rs (torch.Tensor): Rotation matrices of shape (*, 4, 3, 3)
    - Ts (torch.Tensor): Translation vectors of shape (*, 4, 3, 1)
    """
    # Check the shape of the input tensor
    assert E_mat.shape[-2:] == (3, 3), "E_mat must have shape (*, 3, 3)"
    
    # Define the W and Z matrices used in the decomposition
    W = torch.tensor([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]], dtype=E_mat.dtype, device=E_mat.device)
    
    Z = torch.tensor([[0, 1, 0],
                      [-1, 0, 0],
                      [0, 0, 0]], dtype=E_mat.dtype, device=E_mat.device)
    
    # Perform SVD on the essential matrix
    U, _, Vt = torch.svd(E_mat)
    
    # Ensure U and Vt are proper rotation matrices
    if torch.det(U) < 0:
        U[:, :, -1] *= -1
    if torch.det(Vt) < 0:
        Vt[:, -1, :] *= -1
    
    # Compute the possible rotation matrices
    R1 = U @ W @ Vt
    R2 = U @ W.t() @ Vt
    
    # Compute the possible translation vectors
    t = U[:, :, 2].unsqueeze(-1)
    
    # Stack the results to form the output tensors
    Rs = torch.stack([R1, R1, R2, R2], dim=-3)
    Ts = torch.stack([t, -t, t, -t], dim=-3)
    
    return Rs, Ts

