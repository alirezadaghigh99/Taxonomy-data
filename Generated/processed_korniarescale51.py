import torch
import torch.nn.functional as F

def rescale(input, factor, interpolation="bilinear", align_corners=None, antialias=False):
    # Ensure factor is a tuple of two floats
    if isinstance(factor, (int, float)):
        factor = (factor, factor)
    elif isinstance(factor, tuple) and len(factor) == 2:
        factor = tuple(float(f) for f in factor)
    else:
        raise ValueError("Factor must be a float or a tuple of two floats.")
    
    # Calculate the new size
    _, _, height, width = input.shape
    new_height = int(height * factor[0])
    new_width = int(width * factor[1])
    new_size = (new_height, new_width)
    
    # Perform the interpolation
    rescaled_tensor = F.interpolate(input, size=new_size, mode=interpolation, align_corners=align_corners, antialias=antialias)
    
    return rescaled_tensor

