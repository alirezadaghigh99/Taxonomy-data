import torch

def find_homography_lines_dlt(ls1, ls2, weights=None):
    """
    Computes the homography matrix using the DLT formulation for line correspondences.
    
    Args:
        ls1 (torch.Tensor): Tensor of shape (B, N, 2, 2) representing the first set of line segments.
        ls2 (torch.Tensor): Tensor of shape (B, N, 2, 2) representing the second set of line segments.
        weights (torch.Tensor, optional): Tensor of shape (B, N) representing the weights for each correspondence.
        
    Returns:
        torch.Tensor: Tensor of shape (B, 3, 3) representing the computed homography matrices.
    """
    B, N, _, _ = ls1.shape
    
    if weights is None:
        weights = torch.ones((B, N), device=ls1.device, dtype=ls1.dtype)
    
    # Convert line segments to homogeneous line representations (ax + by + c = 0)
    def line_to_homogeneous(p1, p2):
        a = p1[..., 1] - p2[..., 1]
        b = p2[..., 0] - p1[..., 0]
        c = p1[..., 0] * p2[..., 1] - p2[..., 0] * p1[..., 1]
        return torch.stack([a, b, c], dim=-1)
    
    l1 = line_to_homogeneous(ls1[..., 0, :], ls1[..., 1, :])  # Shape (B, N, 3)
    l2 = line_to_homogeneous(ls2[..., 0, :], ls2[..., 1, :])  # Shape (B, N, 3)
    
    # Construct the matrix A for each batch
    A = torch.zeros((B, 2 * N, 9), device=ls1.device, dtype=ls1.dtype)
    
    for i in range(N):
        l1_i = l1[:, i, :]  # Shape (B, 3)
        l2_i = l2[:, i, :]  # Shape (B, 3)
        w_i = weights[:, i].unsqueeze(-1)  # Shape (B, 1)
        
        A[:, 2 * i, 0:3] = w_i * l1_i
        A[:, 2 * i, 6:9] = -w_i * l2_i[:, 0:1] * l1_i
        
        A[:, 2 * i + 1, 3:6] = w_i * l1_i
        A[:, 2 * i + 1, 6:9] = -w_i * l2_i[:, 1:2] * l1_i
    
    # Solve the system using SVD
    U, S, Vh = torch.linalg.svd(A)
    H = Vh[:, -1, :].reshape(B, 3, 3)
    
    return H

