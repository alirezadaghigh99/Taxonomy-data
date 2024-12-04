import torch

def KORNIA_CHECK_LAF(laf: torch.Tensor, raises: bool = False) -> bool:
    """
    Check if the input tensor laf has the shape (B, N, 2, 3).

    Args:
        laf (torch.Tensor): The input tensor to check.
        raises (bool): If True, raise an exception if the shape is invalid.

    Returns:
        bool: True if the shape is valid, False otherwise.
    """
    if laf.shape[-3:] == (2, 3) and len(laf.shape) == 4:
        return True
    else:
        if raises:
            raise ValueError(f"Invalid shape for LAF tensor: {laf.shape}. Expected shape (B, N, 2, 3).")
        return False

