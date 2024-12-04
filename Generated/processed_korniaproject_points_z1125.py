import torch

def project_points_z1(points_in_camera):
    """
    Projects points from the camera frame into the canonical z=1 plane through perspective division.
    
    Args:
    points_in_camera (torch.Tensor): A tensor of shape (..., 3) representing the points to project.
    
    Returns:
    torch.Tensor: A tensor of shape (..., 2) representing the projected points.
    """
    # Ensure the input tensor has the correct shape
    if points_in_camera.shape[-1] != 3:
        raise ValueError("Input tensor must have shape (..., 3)")
    
    # Extract x, y, z coordinates
    x = points_in_camera[..., 0]
    y = points_in_camera[..., 1]
    z = points_in_camera[..., 2]
    
    # Perform perspective division
    x_proj = x / z
    y_proj = y / z
    
    # Stack the projected coordinates to form the output tensor
    projected_points = torch.stack((x_proj, y_proj), dim=-1)
    
    return projected_points

