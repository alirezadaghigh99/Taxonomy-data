import torch
from torch import Tensor
import torch.nn.functional as F

def _perform_padding(image: Tensor) -> tuple[Tensor, int, int]:
    # Get the height and width of the image
    _, _, H, W = image.shape
    
    # Calculate the padding required to make H and W divisible by 16
    h_pad = (16 - H % 16) % 16
    w_pad = (16 - W % 16) % 16
    
    # Calculate padding for each side
    pad_top = h_pad // 2
    pad_bottom = h_pad - pad_top
    pad_left = w_pad // 2
    pad_right = w_pad - pad_left
    
    # Apply padding
    image_padded = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    return image_padded, h_pad, w_pad

