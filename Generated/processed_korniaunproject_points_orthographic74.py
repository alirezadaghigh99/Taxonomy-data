import torch

def unproject_points_orthographic(points_in_camera, extension):
    """
    Unprojects points from the canonical z=1 plane into the camera frame.

    Args:
    points_in_camera (Tensor): A tensor of shape (..., 2) representing the points to unproject.
    extension (Tensor): A tensor of shape (..., 1) representing the extension of the points to unproject.

    Returns:
    Tensor: A tensor of shape (..., 3) representing the unprojected points.
    """
    # Ensure the points_in_camera and extension have compatible shapes
    if points_in_camera.shape[:-1] != extension.shape[:-1]:
        raise ValueError("The shapes of points_in_camera and extension are not compatible.")
    
    # Concatenate the points_in_camera and extension along the last dimension
    unprojected_points = torch.cat((points_in_camera, extension), dim=-1)
    
    return unprojected_points

