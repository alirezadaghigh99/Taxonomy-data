import torch
import numpy as np
import random

def draw_segmentation_masks(image, masks, alpha=0.5, colors=None):
    # Input validation
    if not isinstance(image, torch.Tensor):
        raise TypeError("The image must be a PyTorch tensor.")
    
    if image.dtype not in [torch.uint8, torch.float32, torch.float64]:
        raise ValueError("The image tensor's dtype must be uint8 or a floating point type.")
    
    if image.ndimension() != 3 or image.size(0) != 3:
        raise ValueError("The image tensor must have 3 dimensions and 3 channels (RGB).")
    
    if masks.ndimension() not in [2, 3]:
        raise ValueError("The masks tensor must have 2 or 3 dimensions.")
    
    if masks.dtype != torch.bool:
        raise ValueError("The masks tensor must be of boolean dtype.")
    
    if masks.ndimension() == 2:
        masks = masks.unsqueeze(0)
    
    if image.size(1) != masks.size(1) or image.size(2) != masks.size(2):
        raise ValueError("The spatial dimensions of the masks must match the image.")
    
    num_masks = masks.size(0)
    
    # Generate random colors if not provided
    if colors is None:
        colors = [tuple(np.random.randint(0, 256, 3)) for _ in range(num_masks)]
    elif isinstance(colors, (list, tuple)) and all(isinstance(c, (list, tuple)) and len(c) == 3 for c in colors):
        if len(colors) != num_masks:
            raise ValueError("The number of colors provided must match the number of masks.")
    else:
        raise ValueError("Colors must be a list of RGB tuples or None.")
    
    # Convert image to float for blending
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    
    # Create an overlay image
    overlay = image.clone()
    
    for i in range(num_masks):
        mask = masks[i]
        color = torch.tensor(colors[i], dtype=torch.float32) / 255.0
        for c in range(3):
            overlay[c][mask] = (1 - alpha) * overlay[c][mask] + alpha * color[c]
    
    # Convert back to original dtype
    if image.dtype == torch.uint8:
        overlay = (overlay * 255).byte()
    
    return overlay

