import torch
import numpy as np

def tensor_to_image(tensor, keepdim=True, force_contiguous=True):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    
    if tensor.dim() not in [2, 3, 4]:
        raise ValueError("Input tensor must have 2, 3, or 4 dimensions.")
    
    # Move tensor to CPU if it's on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Make tensor contiguous if required
    if force_contiguous:
        tensor = tensor.contiguous()
    
    # Convert tensor to NumPy array
    np_array = tensor.numpy()
    
    # Handle different shapes
    if tensor.dim() == 2:
        # Shape (H, W)
        return np_array
    elif tensor.dim() == 3:
        # Shape (C, H, W)
        np_array = np.transpose(np_array, (1, 2, 0))  # Convert to (H, W, C)
        return np_array
    elif tensor.dim() == 4:
        # Shape (B, C, H, W)
        np_array = np.transpose(np_array, (0, 2, 3, 1))  # Convert to (B, H, W, C)
        if not keepdim:
            np_array = np.squeeze(np_array, axis=0)  # Remove batch dimension if keepdim is False
        return np_array

