import torch

def project_points(points_3d, camera_matrix):
    """
    Projects 3D points onto a 2D camera plane using the camera matrix.

    Args:
    points_3d (torch.Tensor): Tensor of 3D points with shape (*, 3).
    camera_matrix (torch.Tensor): Tensor of camera matrices with shape (*, 3, 3).

    Returns:
    torch.Tensor: Tensor of 2D camera coordinates with shape (*, 2).
    """
    # Ensure the points are in homogeneous coordinates by adding a column of ones
    ones = torch.ones(points_3d.shape[:-1] + (1,), device=points_3d.device)
    points_3d_homogeneous = torch.cat([points_3d, ones], dim=-1)
    
    # Perform the matrix multiplication to project the points
    points_2d_homogeneous = torch.matmul(camera_matrix, points_3d_homogeneous.unsqueeze(-1)).squeeze(-1)
    
    # Convert from homogeneous coordinates to 2D coordinates
    points_2d = points_2d_homogeneous[..., :2] / points_2d_homogeneous[..., 2:3]
    
    return points_2d

