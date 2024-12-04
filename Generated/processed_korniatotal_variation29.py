import torch

def total_variation(image_tensor, reduction='sum'):
    """
    Computes the Total Variation of an input image tensor.

    Parameters:
    - image_tensor (torch.Tensor): Input image tensor with shape (*, H, W).
    - reduction (str): Specifies whether to return the 'sum' or 'mean' of the output. Default is 'sum'.

    Returns:
    - torch.Tensor: A tensor with shape (*) representing the Total Variation.
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("Input image_tensor must be a torch.Tensor")
    
    if reduction not in ['sum', 'mean']:
        raise ValueError("Reduction must be either 'sum' or 'mean'")
    
    if image_tensor.dim() < 2:
        raise ValueError("Input image_tensor must have at least 2 dimensions (H, W)")

    # Calculate the absolute differences of neighboring pixels along height and width
    diff_h = torch.abs(image_tensor[..., 1:, :] - image_tensor[..., :-1, :])
    diff_w = torch.abs(image_tensor[..., :, 1:] - image_tensor[..., :, :-1])
    
    # Sum the differences
    total_variation = diff_h.sum(dim=(-2, -1)) + diff_w.sum(dim=(-2, -1))
    
    if reduction == 'mean':
        total_variation = total_variation / (image_tensor.size(-2) * image_tensor.size(-1))
    
    return total_variation

