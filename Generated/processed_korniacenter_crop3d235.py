import torch
import torch.nn.functional as F

def center_crop3d(tensor, size, interpolation='trilinear', align_corners=False):
    """
    Crops a 3D volume tensor to the center region based on the specified dimensions.

    Parameters:
    tensor (torch.Tensor): Input tensor of shape (B, C, D, H, W).
    size (tuple): Desired output dimensions (depth, height, width).
    interpolation (str, optional): Interpolation mode. Default is 'trilinear'.
    align_corners (bool, optional): If True, the corner pixels of the input and output tensors are aligned. Default is False.

    Returns:
    torch.Tensor: Cropped 3D volume tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    if len(tensor.shape) != 5:
        raise ValueError("Input tensor must have shape (B, C, D, H, W)")
    
    if not isinstance(size, tuple) or len(size) != 3:
        raise ValueError("Size must be a tuple of three integers (depth, height, width)")
    
    B, C, D, H, W = tensor.shape
    target_d, target_h, target_w = size
    
    if target_d > D or target_h > H or target_w > W:
        raise ValueError("Target size must be less than or equal to the input size in all dimensions")
    
    # Calculate the starting indices for cropping
    start_d = (D - target_d) // 2
    start_h = (H - target_h) // 2
    start_w = (W - target_w) // 2
    
    # Crop the tensor
    cropped_tensor = tensor[:, :, start_d:start_d + target_d, start_h:start_h + target_h, start_w:start_w + target_w]
    
    # If interpolation is needed, resize the cropped tensor
    if (target_d, target_h, target_w) != (cropped_tensor.shape[2], cropped_tensor.shape[3], cropped_tensor.shape[4]):
        cropped_tensor = F.interpolate(cropped_tensor, size=size, mode=interpolation, align_corners=align_corners)
    
    return cropped_tensor

