import torch

def laf_from_center_scale_ori(xy, scale=None, ori=None):
    # Check the shape of the input tensor xy
    if xy.ndimension() != 3 or xy.size(2) != 2:
        raise ValueError("Input tensor xy must have shape (B, N, 2)")
    
    # Initialize device and data type
    device = xy.device
    dtype = xy.dtype
    
    # Calculate batch size B and number of keypoints N
    B, N, _ = xy.shape
    
    # If scale is not provided, set it to ones tensor of the appropriate shape
    if scale is None:
        scale = torch.ones((B, N, 1, 1), device=device, dtype=dtype)
    else:
        if scale.ndimension() != 4 or scale.size(2) != 1 or scale.size(3) != 1:
            raise ValueError("Scale tensor must have shape (B, N, 1, 1)")
    
    # If orientation is not provided, set it to zeros tensor of the appropriate shape
    if ori is None:
        ori = torch.zeros((B, N, 1), device=device, dtype=dtype)
    else:
        if ori.ndimension() != 3 or ori.size(2) != 1:
            raise ValueError("Orientation tensor must have shape (B, N, 1)")
    
    # Calculate the rotation matrix based on the orientation
    cos_ori = torch.cos(ori)
    sin_ori = torch.sin(ori)
    rotation_matrix = torch.cat([cos_ori, -sin_ori, sin_ori, cos_ori], dim=2).view(B, N, 2, 2)
    
    # Scale the rotation matrix
    scaled_rotation_matrix = rotation_matrix * scale
    
    # Concatenate the scaled rotation matrix with the keypoint centers
    LAF = torch.cat([scaled_rotation_matrix, xy.unsqueeze(2)], dim=3)
    
    return LAF

