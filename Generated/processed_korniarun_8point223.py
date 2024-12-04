import torch

def normalize_points(points):
    """ Normalize points to improve numerical stability. """
    mean = points.mean(dim=1, keepdim=True)
    std = points.std(dim=1, keepdim=True)
    normalized_points = (points - mean) / std
    T = torch.eye(3, device=points.device).unsqueeze(0).repeat(points.shape[0], 1, 1)
    T[:, 0, 0] = 1 / std[:, 0, 0]
    T[:, 1, 1] = 1 / std[:, 0, 1]
    T[:, 0, 2] = -mean[:, 0, 0] / std[:, 0, 0]
    T[:, 1, 2] = -mean[:, 0, 1] / std[:, 0, 1]
    return normalized_points, T

def run_8point(points1, points2, weights):
    """
    Compute the fundamental matrix using the DLT formulation with weighted least squares.
    
    Args:
        points1: A set of points in the first image with a tensor shape (B, N, 2), N>=8.
        points2: A set of points in the second image with a tensor shape (B, N, 2), N>=8.
        weights: Tensor containing the weights per point correspondence with a shape of (B, N).
    
    Returns:
        The computed fundamental matrix with shape (B, 3, 3).
    """
    B, N, _ = points1.shape
    
    # Normalize points
    points1_normalized, T1 = normalize_points(points1)
    points2_normalized, T2 = normalize_points(points2)
    
    # Construct matrix A
    x1, y1 = points1_normalized[..., 0], points1_normalized[..., 1]
    x2, y2 = points2_normalized[..., 0], points2_normalized[..., 1]
    
    A = torch.stack([
        x2 * x1, x2 * y1, x2,
        y2 * x1, y2 * y1, y2,
        x1, y1, torch.ones_like(x1)
    ], dim=-1)  # Shape: (B, N, 9)
    
    # Apply weights
    W = torch.diag_embed(weights)  # Shape: (B, N, N)
    AW = torch.matmul(W, A)  # Shape: (B, N, 9)
    
    # Solve the weighted least squares problem
    _, _, V = torch.svd(AW)
    F = V[..., -1].view(B, 3, 3)  # Shape: (B, 3, 3)
    
    # Enforce rank-2 constraint
    U, S, Vt = torch.svd(F)
    S[..., -1] = 0
    F_rank2 = torch.matmul(U, torch.matmul(torch.diag_embed(S), Vt))
    
    # Denormalize the fundamental matrix
    F_denormalized = torch.matmul(T2.transpose(1, 2), torch.matmul(F_rank2, T1))
    
    return F_denormalized

