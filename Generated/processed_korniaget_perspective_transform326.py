import torch

def get_perspective_transform(points_src, points_dst):
    """
    Calculate a perspective transform from four pairs of the corresponding points using DLT.

    Args:
        points_src: coordinates of quadrangle vertices in the source image with shape (B, 4, 2).
        points_dst: coordinates of the corresponding quadrangle vertices in the destination image with shape (B, 4, 2).

    Returns:
        the perspective transformation with shape (B, 3, 3).
    """
    assert points_src.shape == points_dst.shape, "Source and destination points must have the same shape"
    assert points_src.shape[1:] == (4, 2), "Each set of points must have shape (4, 2)"
    
    batch_size = points_src.shape[0]
    A = torch.zeros((batch_size, 8, 9), dtype=points_src.dtype, device=points_src.device)
    
    for i in range(4):
        x_src, y_src = points_src[:, i, 0], points_src[:, i, 1]
        x_dst, y_dst = points_dst[:, i, 0], points_dst[:, i, 1]
        
        A[:, 2*i, 0:3] = torch.stack([x_src, y_src, torch.ones_like(x_src)], dim=1)
        A[:, 2*i, 6:9] = -x_dst.unsqueeze(1) * torch.stack([x_src, y_src, torch.ones_like(x_src)], dim=1)
        
        A[:, 2*i+1, 3:6] = torch.stack([x_src, y_src, torch.ones_like(x_src)], dim=1)
        A[:, 2*i+1, 6:9] = -y_dst.unsqueeze(1) * torch.stack([x_src, y_src, torch.ones_like(x_src)], dim=1)
    
    # Solve Ah = 0 using SVD
    U, S, V = torch.svd(A)
    H = V[:, -1, :].reshape(batch_size, 3, 3)
    
    return H

