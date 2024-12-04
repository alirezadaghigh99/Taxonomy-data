import torch
from torch import Tensor

def KORNIA_CHECK_SHAPE(input: Tensor, shape: list):
    # This is a placeholder for the actual shape checking function.
    # You can implement this function based on your specific requirements.
    pass

def normalize_kernel2d(input: Tensor) -> Tensor:
    r"""Normalize both derivative and smoothing kernel."""
    KORNIA_CHECK_SHAPE(input, ["*", "H", "W"])
    
    # Ensure the input is a floating-point tensor
    if not input.is_floating_point():
        input = input.float()
    
    # Compute the sum of all elements in the tensor
    sum_elements = input.sum()
    
    # Normalize the tensor by dividing each element by the sum
    normalized_tensor = input / sum_elements
    
    return normalized_tensor

