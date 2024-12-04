import torch
import torch.nn.functional as F

def resize(input_tensor, output_size, interpolation='bilinear', align_corners=False, side='both', antialias=False):
    """
    Resizes an input torch.Tensor to a specified size using various parameters.

    Parameters:
    - input_tensor (torch.Tensor): The input tensor to be resized.
    - output_size (tuple): The desired output size (height, width).
    - interpolation (str): The interpolation method ('bilinear', 'nearest', etc.). Default is 'bilinear'.
    - align_corners (bool): Flag for alignment. Default is False.
    - side (str): Side for resizing ('both', 'height', 'width'). Default is 'both'.
    - antialias (bool): Flag for antialiasing. Default is False.

    Returns:
    - torch.Tensor: The resized tensor.
    """
    # Validate input tensor
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("input_tensor must be a torch.Tensor")
    
    if len(input_tensor.shape) != 4:
        raise ValueError("input_tensor must be a 4D tensor (batch_size, channels, height, width)")

    # Validate output size
    if not (isinstance(output_size, tuple) and len(output_size) == 2):
        raise ValueError("output_size must be a tuple of (height, width)")

    # Validate interpolation method
    valid_interpolations = ['bilinear', 'nearest', 'bicubic', 'trilinear', 'area']
    if interpolation not in valid_interpolations:
        raise ValueError(f"Invalid interpolation method. Choose from {valid_interpolations}")

    # Validate side
    valid_sides = ['both', 'height', 'width']
    if side not in valid_sides:
        raise ValueError(f"Invalid side. Choose from {valid_sides}")

    # Determine the new size based on the side parameter
    if side == 'both':
        new_size = output_size
    elif side == 'height':
        new_size = (output_size[0], input_tensor.shape[3])
    elif side == 'width':
        new_size = (input_tensor.shape[2], output_size[1])

    # Resize the tensor
    resized_tensor = F.interpolate(input_tensor, size=new_size, mode=interpolation, align_corners=align_corners, antialias=antialias)

    return resized_tensor

