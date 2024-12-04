import torch

def depth_to_3d(depth, camera_matrix, normalize=False):
    """
    Convert depth map to 3D point cloud.

    Args:
        depth (torch.Tensor): Depth tensor of shape (B, 1, H, W).
        camera_matrix (torch.Tensor): Camera intrinsics tensor of shape (B, 3, 3).
        normalize (bool): Whether to normalize the 3D points.

    Returns:
        torch.Tensor: 3D points tensor of shape (B, 3, H, W).
    """
    # Validate input types
    if not isinstance(depth, torch.Tensor):
        raise TypeError("Depth must be a torch.Tensor")
    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError("Camera matrix must be a torch.Tensor")
    
    # Validate input shapes
    if depth.ndimension() != 4 or depth.size(1) != 1:
        raise ValueError("Depth tensor must have shape (B, 1, H, W)")
    if camera_matrix.ndimension() != 3 or camera_matrix.size(1) != 3 or camera_matrix.size(2) != 3:
        raise ValueError("Camera matrix tensor must have shape (B, 3, 3)")
    
    B, _, H, W = depth.shape

    # Create a mesh grid of pixel coordinates
    y, x = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device))
    y = y.float()
    x = x.float()

    # Expand the grid to match the batch size
    x = x.unsqueeze(0).expand(B, -1, -1)
    y = y.unsqueeze(0).expand(B, -1, -1)

    # Extract camera intrinsics
    fx = camera_matrix[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
    fy = camera_matrix[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = camera_matrix[:, 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = camera_matrix[:, 1, 2].unsqueeze(-1).unsqueeze(-1)

    # Compute the 3D coordinates
    Z = depth.squeeze(1)
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy

    # Stack the coordinates to get the final 3D points
    points_3d = torch.stack((X, Y, Z), dim=1)

    if normalize:
        # Normalize the 3D points
        norm = torch.norm(points_3d, dim=1, keepdim=True)
        points_3d = points_3d / norm

    return points_3d

