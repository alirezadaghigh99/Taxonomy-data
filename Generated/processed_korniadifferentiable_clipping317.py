import torch
import torch.nn.functional as F

def differentiable_clipping(input, min_val=None, max_val=None, scale=0.02):
    """
    Implements a differentiable and soft approximation of the clipping operation.

    Args:
        input (Tensor): Input tensor of any shape.
        min_val (Optional[float]): Minimum value.
        max_val (Optional[float]): Maximum value.
        scale (float): Scale value. Default 0.02.

    Returns:
        output (Tensor): Clipped output tensor of the same shape as the input tensor.
    """
    if min_val is not None:
        min_val = torch.tensor(min_val, dtype=input.dtype, device=input.device)
    if max_val is not None:
        max_val = torch.tensor(max_val, dtype=input.dtype, device=input.device)
    
    if min_val is not None and max_val is not None:
        # Scale and shift tanh to approximate clipping between min_val and max_val
        output = (max_val - min_val) * 0.5 * (torch.tanh(scale * (input - (min_val + max_val) * 0.5)) + 1) + min_val
    elif min_val is not None:
        # Scale and shift tanh to approximate clipping with only min_val
        output = (input - min_val) * 0.5 * (torch.tanh(scale * (input - min_val)) + 1) + min_val
    elif max_val is not None:
        # Scale and shift tanh to approximate clipping with only max_val
        output = (max_val - input) * 0.5 * (torch.tanh(scale * (max_val - input)) + 1) + input
    else:
        # No clipping needed
        output = input

    return output