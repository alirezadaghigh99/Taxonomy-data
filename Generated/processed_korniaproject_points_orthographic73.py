import torch

def project_points_orthographic(points_in_camera):
    """
    Project one or more points from the camera frame into the canonical z=1 plane through orthographic projection.

    Args:
        points_in_camera: Tensor representing the points to project. Shape should be (..., 3).

    Returns:
        Tensor representing the projected points. Shape will be (..., 2).
    """
    # Ensure the input is a tensor
    points_in_camera = torch.tensor(points_in_camera, dtype=torch.float32)
    
    # Check if the last dimension is 3 (x, y, z)
    if points_in_camera.shape[-1] != 3:
        raise ValueError("The last dimension of points_in_camera must be 3 (representing x, y, z coordinates).")
    
    # Perform orthographic projection by dropping the z-coordinate
    projected_points = points_in_camera[..., :2]
    
    return projected_points

