import torch
import torch.nn.functional as F

def remap(image, map_x, map_y, mode='bilinear', padding_mode='zeros', align_corners=False, normalized_coordinates=False):
    """
    Apply a generic geometrical transformation to an image tensor.

    Args:
        image: the tensor to remap with shape (B, C, H, W).
        map_x: the flow in the x-direction in pixel coordinates with shape (B, H, W).
        map_y: the flow in the y-direction in pixel coordinates with shape (B, H, W).
        mode: interpolation mode to calculate output values 'bilinear' | 'nearest'.
        padding_mode: padding mode for outside grid values 'zeros' | 'border' | 'reflection'.
        align_corners: mode for grid_generation.
        normalized_coordinates: whether the input coordinates are normalized in the range of [-1, 1].

    Returns:
        the warped tensor with same shape as the input grid maps.
    """
    B, C, H, W = image.shape

    # Stack map_x and map_y to create the grid
    grid = torch.stack((map_x, map_y), dim=-1)  # Shape: (B, H, W, 2)

    if not normalized_coordinates:
        # Normalize the grid to the range [-1, 1]
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0

    # Perform the remapping using grid_sample
    remapped_image = F.grid_sample(image, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    return remapped_image

