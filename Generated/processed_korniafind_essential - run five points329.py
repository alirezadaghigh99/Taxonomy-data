import torch

def run_5point(points1, points2, weights=None):
    """
    Computes the essential matrix using Nister's 5-point algorithm.
    
    Args:
        points1 (torch.Tensor): A tensor of shape (B, 5, 2) representing the calibrated points from the first image.
        points2 (torch.Tensor): A tensor of shape (B, 5, 2) representing the calibrated points from the second image.
        weights (torch.Tensor, optional): A tensor of shape (B, 5) representing the weights for each point pair.
        
    Returns:
        torch.Tensor: A tensor of shape (B, 3, 3) representing the essential matrix for each batch.
    """
    B, N, _ = points1.shape
    assert N == 5, "The 5-point algorithm requires exactly 5 point correspondences."
    assert points1.shape == points2.shape, "points1 and points2 must have the same shape."
    if weights is not None:
        assert weights.shape == (B, N), "weights must have shape (B, 5)."
    
    # Construct the linear system
    A = torch.zeros((B, N, 9), dtype=points1.dtype, device=points1.device)
    for i in range(N):
        x1, y1 = points1[:, i, 0], points1[:, i, 1]
        x2, y2 = points2[:, i, 0], points2[:, i, 1]
        A[:, i, :] = torch.stack([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, torch.ones_like(x1)], dim=-1)
    
    if weights is not None:
        A *= weights.unsqueeze(-1)
    
    # Solve the linear system using SVD
    _, _, V = torch.svd(A)
    E_candidates = V[:, -4:, :].reshape(B, 4, 3, 3)
    
    # Solve the polynomial constraints to find the correct E
    # This part is simplified and may need a more robust implementation
    E = E_candidates[:, 0, :, :]  # Placeholder: select the first candidate
    
    return E

