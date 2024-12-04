import torch
import math

def get_rotation_matrix2d(center, angle, scale):
    """
    Calculate the affine matrix of 2D rotation.

    Args:
        center (Tensor): Center of the rotation in the source image with shape (B, 2).
        angle (Tensor): Rotation angle in degrees with shape (B).
        scale (Tensor): Scale factor for x, y scaling with shape (B, 2).

    Returns:
        Tensor: Affine matrix of 2D rotation with shape (B, 2, 3).
    """
    assert center.shape[1] == 2, "Center should have shape (B, 2)"
    assert angle.dim() == 1, "Angle should have shape (B)"
    assert scale.shape[1] == 2, "Scale should have shape (B, 2)"
    assert center.shape[0] == angle.shape[0] == scale.shape[0], "Batch size should be the same for all inputs"

    B = center.shape[0]
    angle_rad = angle * math.pi / 180.0

    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    scale_x = scale[:, 0]
    scale_y = scale[:, 1]

    M = torch.zeros((B, 2, 3), dtype=center.dtype, device=center.device)

    M[:, 0, 0] = cos_a * scale_x
    M[:, 0, 1] = -sin_a * scale_y
    M[:, 1, 0] = sin_a * scale_x
    M[:, 1, 1] = cos_a * scale_y

    M[:, 0, 2] = center[:, 0] - M[:, 0, 0] * center[:, 0] - M[:, 0, 1] * center[:, 1]
    M[:, 1, 2] = center[:, 1] - M[:, 1, 0] * center[:, 0] - M[:, 1, 1] * center[:, 1]

    return M

