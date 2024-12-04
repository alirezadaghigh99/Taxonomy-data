import torch

def rgb_to_rgba(image: torch.Tensor, alpha_val: float or torch.Tensor) -> torch.Tensor:
    """
    Convert an image from RGB to RGBA.

    Args:
        image (torch.Tensor): RGB Image to be converted to RGBA of shape :math:`(*,3,H,W)`.
        alpha_val (float or torch.Tensor): A float number for the alpha value or a tensor
          of shape :math:`(*,1,H,W)`.

    Returns:
        torch.Tensor: RGBA version of the image with shape :math:`(*,4,H,W)`.
    """
    if image.shape[-4] != 3:
        raise ValueError("Input image must have 3 channels in the last dimension")

    if isinstance(alpha_val, float):
        alpha_channel = torch.full_like(image[:, :1, :, :], alpha_val)
    elif isinstance(alpha_val, torch.Tensor):
        if alpha_val.shape != image[:, :1, :, :].shape:
            raise ValueError("Alpha tensor must have the same shape as the input image's single channel")
        alpha_channel = alpha_val
    else:
        raise TypeError("alpha_val must be either a float or a torch.Tensor")

    rgba_image = torch.cat((image, alpha_channel), dim=-4)
    return rgba_image

