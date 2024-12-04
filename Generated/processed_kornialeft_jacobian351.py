import torch

def left_jacobian(vec):
    """
    Compute the left Jacobian of SO(3) for a batch of vectors.
    
    Args:
        vec (torch.Tensor): A tensor of shape (B, 3) where B is the batch size.
        
    Returns:
        torch.Tensor: A tensor of shape (B, 3, 3) representing the left Jacobian matrices.
    """
    B = vec.shape[0]
    theta = torch.norm(vec, dim=1, keepdim=True)  # Shape (B, 1)
    theta_sq = theta ** 2
    theta_cube = theta ** 3

    # Small angle approximation for theta near zero
    small_angle = theta < 1e-6

    # Compute the skew-symmetric matrix of vec
    def skew_symmetric(v):
        zero = torch.zeros_like(v[:, 0])
        return torch.stack([
            torch.stack([zero, -v[:, 2], v[:, 1]], dim=1),
            torch.stack([v[:, 2], zero, -v[:, 0]], dim=1),
            torch.stack([-v[:, 1], v[:, 0], zero], dim=1)
        ], dim=1)

    K = skew_symmetric(vec)  # Shape (B, 3, 3)

    # Compute the left Jacobian using the series expansion
    I = torch.eye(3, device=vec.device).unsqueeze(0).repeat(B, 1, 1)  # Shape (B, 3, 3)
    A = I + (1 - torch.cos(theta)) / theta_sq * K + (theta - torch.sin(theta)) / theta_cube * torch.bmm(K, K)

    # Handle small angles separately to avoid division by zero
    if small_angle.any():
        A[small_angle.squeeze()] = I[small_angle.squeeze()] + 0.5 * K[small_angle.squeeze()]

    return A

