import torch
from torch import Tensor

def laf_from_three_points(threepts: Tensor) -> Tensor:
    """Convert three points to local affine frame.

    Order is (0,0), (0, 1), (1, 0).

    Args:
        threepts: :math:`(B, N, 2, 3)`.

    Returns:
        laf :math:`(B, N, 2, 3)`.
    """
    # Extract the points
    p0 = threepts[:, :, :, 0]  # (B, N, 2)
    p1 = threepts[:, :, :, 1]  # (B, N, 2)
    p2 = threepts[:, :, :, 2]  # (B, N, 2)

    # The origin of the local affine frame is p0
    origin = p0

    # The x-axis direction is defined by the vector from p0 to p2
    x_axis = p2 - p0

    # The y-axis direction is defined by the vector from p0 to p1
    y_axis = p1 - p0

    # Stack the origin, x_axis, and y_axis to form the LAF
    laf = torch.stack([origin, x_axis, y_axis], dim=-1)  # (B, N, 2, 3)

    return laf