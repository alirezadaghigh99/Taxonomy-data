import torch
from typing import Tuple

def rgb_to_yuv420(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if image.dim() != 4 or image.size(1) != 3:
        raise ValueError("Input image must have shape (*, 3, H, W)")
    
    _, _, H, W = image.shape
    
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError("Height and width of the input image must be even numbers")
    
    # RGB to YUV conversion matrix
    rgb_to_yuv_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51499, -0.10001]
    ], dtype=image.dtype, device=image.device)
    
    # Reshape image to (N, H, W, 3) for matrix multiplication
    image = image.permute(0, 2, 3, 1)
    
    # Convert RGB to YUV
    yuv_image = torch.tensordot(image, rgb_to_yuv_matrix, dims=([3], [1]))
    
    # Split Y, U, V channels
    Y = yuv_image[..., 0]
    U = yuv_image[..., 1]
    V = yuv_image[..., 2]
    
    # Reshape Y to (N, 1, H, W)
    Y = Y.unsqueeze(1)
    
    # Subsample U and V channels
    U_subsampled = U.unfold(1, 2, 2).unfold(2, 2, 2).mean(dim=(-1, -2))
    V_subsampled = V.unfold(1, 2, 2).unfold(2, 2, 2).mean(dim=(-1, -2))
    
    # Stack U and V channels
    UV = torch.stack((U_subsampled, V_subsampled), dim=1)
    
    return Y, UV

