import torch

def deg2rad(tensor):
    """
    Converts angles from degrees to radians.

    Args:
        tensor: Tensor of arbitrary shape.

    Returns:
        Tensor with same shape as input, with angles converted to radians.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(tensor)}")
    
    return tensor * (torch.pi / 180)

