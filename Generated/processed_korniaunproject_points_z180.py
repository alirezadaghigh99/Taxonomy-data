import torch

def unproject_points_z1(points_in_cam_canonical, extension=None):
    """
    Unprojects points from the canonical z=1 plane into the camera frame.

    Args:
        points_in_cam_canonical (Tensor): Points to unproject with shape (..., 2).
        extension (Tensor, optional): Extension (depth) of the points with shape (..., 1). Defaults to None.

    Returns:
        Tensor: Unprojected points with shape (..., 3).
    """
    # Ensure points_in_cam_canonical has the correct shape
    if points_in_cam_canonical.shape[-1] != 2:
        raise ValueError("points_in_cam_canonical must have shape (..., 2)")

    # If extension is not provided, assume depth of 1 for all points
    if extension is None:
        extension = torch.ones(points_in_cam_canonical.shape[:-1] + (1,), device=points_in_cam_canonical.device)

    # Ensure extension has the correct shape
    if extension.shape[-1] != 1:
        raise ValueError("extension must have shape (..., 1)")

    # Concatenate the points with the extension to form the unprojected points
    unprojected_points = torch.cat((points_in_cam_canonical, extension), dim=-1)

    return unprojected_points

