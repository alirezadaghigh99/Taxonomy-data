import torch
from torch import Tensor

def projection_from_KRt(K: Tensor, R: Tensor, t: Tensor) -> Tensor:
    """
    Get the projection matrix P from K, R and t.

    This function estimates the projection matrix by solving the following equation: P = K * [R|t].

    Args:
       K: the camera matrix with the intrinsics with shape (B, 3, 3).
       R: The rotation matrix with shape (B, 3, 3).
       t: The translation vector with shape (B, 3, 1).

    Returns:
       The projection matrix P with shape (B, 4, 4).
    """
    if K.shape[-2:] != (3, 3):
        raise AssertionError(K.shape)
    if R.shape[-2:] != (3, 3):
        raise AssertionError(R.shape)
    if t.shape[-2:] != (3, 1):
        raise AssertionError(t.shape)
    if not len(K.shape) == len(R.shape) == len(t.shape):
        raise AssertionError

    B = K.shape[0]

    # Concatenate R and t to form [R|t] with shape (B, 3, 4)
    Rt = torch.cat((R, t), dim=-1)

    # Add a row [0, 0, 0, 1] to make it a 4x4 matrix
    Rt_homogeneous = torch.cat((Rt, torch.tensor([[[0, 0, 0, 1]]], dtype=Rt.dtype, device=Rt.device).repeat(B, 1, 1)), dim=1)

    # Create a 4x4 K matrix by adding a row [0, 0, 0, 1]
    K_homogeneous = torch.cat((K, torch.zeros((B, 3, 1), dtype=K.dtype, device=K.device)), dim=-1)
    K_homogeneous = torch.cat((K_homogeneous, torch.tensor([[[0, 0, 0, 1]]], dtype=K.dtype, device=K.device).repeat(B, 1, 1)), dim=1)

    # Compute the projection matrix P
    P = torch.matmul(K_homogeneous, Rt_homogeneous)

    return P