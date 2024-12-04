import torch

def denormalize_laf(LAF, images):
    """
    De-normalize LAFs from scale to image scale. The convention is that center of 5-pixel image (coordinates
    from 0 to 4) is 2, and not 2.5.

    Args:
        LAF: :math:`(B, N, 2, 3)`
        images: :math:`(B, CH, H, W)`

    Returns:
        the denormalized LAF: :math:`(B, N, 2, 3)`, scale in pixels
    """
    B, N, _, _ = LAF.size()
    _, _, H, W = images.size()
    MIN_SIZE = min(H - 1, W - 1)

    # Create a copy of LAF to avoid modifying the original tensor
    denormalized_LAF = LAF.clone()

    # Apply the transformation
    denormalized_LAF[:, :, 0, 0] *= MIN_SIZE
    denormalized_LAF[:, :, 0, 1] *= MIN_SIZE
    denormalized_LAF[:, :, 1, 0] *= MIN_SIZE
    denormalized_LAF[:, :, 1, 1] *= MIN_SIZE

    denormalized_LAF[:, :, 0, 2] *= (W - 1)
    denormalized_LAF[:, :, 1, 2] *= (H - 1)

    return denormalized_LAF

