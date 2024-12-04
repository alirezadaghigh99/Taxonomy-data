import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

def elastic_transform(img, displacement, interpolation=InterpolationMode.BILINEAR, fill=None):
    """
    Apply elastic transformation to a tensor image.

    Parameters:
    - img (PIL Image or Tensor): Input image.
    - displacement (Tensor): Displacement field tensor.
    - interpolation (InterpolationMode): Interpolation mode.
    - fill (list of floats, optional): Fill value for the areas outside the transformed image.

    Returns:
    - Transformed tensor image.
    """
    if isinstance(img, Image.Image):
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    if not isinstance(img, torch.Tensor):
        raise TypeError("img should be a PIL Image or a Tensor")

    if not isinstance(displacement, torch.Tensor):
        raise TypeError("displacement should be a Tensor")

    if fill is not None and not isinstance(fill, (list, tuple)):
        raise TypeError("fill should be a list or tuple of floats")

    # Ensure the displacement field has the same height and width as the image
    if displacement.shape[1:] != img.shape[1:]:
        raise ValueError("displacement field must have the same height and width as the image")

    # Create a meshgrid for the original image coordinates
    height, width = img.shape[1], img.shape[2]
    grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width))
    grid = torch.stack((grid_x, grid_y), 2).float()

    # Add the displacement field to the grid
    grid = grid + displacement.permute(1, 2, 0)

    # Normalize the grid to the range [-1, 1]
    grid[:, :, 0] = 2.0 * grid[:, :, 0] / (width - 1) - 1.0
    grid[:, :, 1] = 2.0 * grid[:, :, 1] / (height - 1) - 1.0

    # Apply the grid sampling
    grid = grid.unsqueeze(0)
    img = img.unsqueeze(0)
    transformed_img = F.grid_sample(img, grid, mode=interpolation.value, padding_mode='zeros', align_corners=True)

    if fill is not None:
        fill_tensor = torch.tensor(fill).view(1, -1, 1, 1)
        mask = (grid < -1) | (grid > 1)
        mask = mask.any(dim=-1, keepdim=True).permute(0, 3, 1, 2)
        transformed_img = transformed_img * ~mask + fill_tensor * mask

    return transformed_img.squeeze(0)

