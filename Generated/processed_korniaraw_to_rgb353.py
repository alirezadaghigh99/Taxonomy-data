import torch
import torch.nn.functional as F

def raw_to_rgb(image: torch.Tensor, cfa: str) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}.")
    
    if image.dim() < 4 or image.size(-3) != 1:
        raise ValueError(f"Input size must have a shape of (*, 1, H, W). Got {image.shape}.")
    
    _, _, H, W = image.shape
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(f"Input H&W must be evenly divisible by 2. Got {image.shape}.")
    
    if cfa not in ['BG', 'GB', 'RG', 'GR']:
        raise ValueError(f"Invalid CFA configuration. Got {cfa}. Expected one of ['BG', 'GB', 'RG', 'GR'].")
    
    # Define the convolution kernels for bilinear interpolation
    kernel = torch.tensor([[0.25, 0.5, 0.25],
                           [0.5, 1.0, 0.5],
                           [0.25, 0.5, 0.25]], dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    
    # Initialize the RGB channels
    R = torch.zeros_like(image)
    G = torch.zeros_like(image)
    B = torch.zeros_like(image)
    
    # Extract the raw Bayer pattern
    if cfa == 'BG':
        B[:, :, 0::2, 0::2] = image[:, :, 0::2, 0::2]
        G[:, :, 0::2, 1::2] = image[:, :, 0::2, 1::2]
        G[:, :, 1::2, 0::2] = image[:, :, 1::2, 0::2]
        R[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
    elif cfa == 'GB':
        G[:, :, 0::2, 0::2] = image[:, :, 0::2, 0::2]
        B[:, :, 0::2, 1::2] = image[:, :, 0::2, 1::2]
        R[:, :, 1::2, 0::2] = image[:, :, 1::2, 0::2]
        G[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
    elif cfa == 'RG':
        R[:, :, 0::2, 0::2] = image[:, :, 0::2, 0::2]
        G[:, :, 0::2, 1::2] = image[:, :, 0::2, 1::2]
        G[:, :, 1::2, 0::2] = image[:, :, 1::2, 0::2]
        B[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
    elif cfa == 'GR':
        G[:, :, 0::2, 0::2] = image[:, :, 0::2, 0::2]
        R[:, :, 0::2, 1::2] = image[:, :, 0::2, 1::2]
        B[:, :, 1::2, 0::2] = image[:, :, 1::2, 0::2]
        G[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
    
    # Perform bilinear interpolation for R and B channels
    R = F.conv2d(R, kernel, padding=1, groups=1)
    B = F.conv2d(B, kernel, padding=1, groups=1)
    
    # Combine the channels into an RGB image
    rgb_image = torch.cat([R, G, B], dim=-3)
    
    return rgb_image

