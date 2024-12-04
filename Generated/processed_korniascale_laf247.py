import torch

def scale_laf(laf, scale_coef):
    """
    Scales the region part of a Local Affine Frame (LAF) by a scale coefficient.
    
    Args:
        laf (torch.Tensor): Tensor of shape (B, N, 2, 3) representing the LAFs.
        scale_coef (float or torch.Tensor): Scale coefficient to scale the region part.
        
    Returns:
        torch.Tensor: Scaled LAF of the same shape as input.
        
    Raises:
        TypeError: If scale_coef is neither a float nor a tensor.
    """
    if not isinstance(scale_coef, (float, torch.Tensor)):
        raise TypeError("scale_coef must be either a float or a torch.Tensor")
    
    if isinstance(scale_coef, float):
        scale_coef = torch.tensor(scale_coef, dtype=laf.dtype, device=laf.device)
    
    # Ensure scale_coef is broadcastable to the shape (B, N, 1, 1)
    scale_coef = scale_coef.view(-1, 1, 1, 1)
    
    # Extract the affine part (2x2) and the translation part (2x1)
    affine_part = laf[..., :2]
    translation_part = laf[..., 2:]
    
    # Scale the affine part
    scaled_affine_part = affine_part * scale_coef
    
    # Combine the scaled affine part with the unchanged translation part
    scaled_laf = torch.cat([scaled_affine_part, translation_part], dim=-1)
    
    return scaled_laf

