import torch
import math

def get_projective_transform(center, angles, scales):
    # Validate input shapes
    assert center.shape[-1] == 3, "center must have shape (B, 3)"
    assert angles.shape[-1] == 3, "angles must have shape (B, 3)"
    assert scales.shape[-1] == 3, "scales must have shape (B, 3)"
    assert center.shape[0] == angles.shape[0] == scales.shape[0], "center, angles, and scales must have the same batch size"
    
    # Validate input types
    assert center.device == angles.device == scales.device, "center, angles, and scales must be on the same device"
    assert center.dtype == angles.dtype == scales.dtype, "center, angles, and scales must have the same dtype"
    
    B = center.shape[0]
    
    # Convert angles from degrees to radians
    angles = angles * (math.pi / 180.0)
    
    # Compute rotation matrices for each axis
    cos_x = torch.cos(angles[:, 0])
    sin_x = torch.sin(angles[:, 0])
    cos_y = torch.cos(angles[:, 1])
    sin_y = torch.sin(angles[:, 1])
    cos_z = torch.cos(angles[:, 2])
    sin_z = torch.sin(angles[:, 2])
    
    # Rotation matrix around x-axis
    R_x = torch.stack([
        torch.stack([torch.ones(B, device=center.device, dtype=center.dtype), torch.zeros(B, device=center.device, dtype=center.dtype), torch.zeros(B, device=center.device, dtype=center.dtype)], dim=1),
        torch.stack([torch.zeros(B, device=center.device, dtype=center.dtype), cos_x, -sin_x], dim=1),
        torch.stack([torch.zeros(B, device=center.device, dtype=center.dtype), sin_x, cos_x], dim=1)
    ], dim=1)
    
    # Rotation matrix around y-axis
    R_y = torch.stack([
        torch.stack([cos_y, torch.zeros(B, device=center.device, dtype=center.dtype), sin_y], dim=1),
        torch.stack([torch.zeros(B, device=center.device, dtype=center.dtype), torch.ones(B, device=center.device, dtype=center.dtype), torch.zeros(B, device=center.device, dtype=center.dtype)], dim=1),
        torch.stack([-sin_y, torch.zeros(B, device=center.device, dtype=center.dtype), cos_y], dim=1)
    ], dim=1)
    
    # Rotation matrix around z-axis
    R_z = torch.stack([
        torch.stack([cos_z, -sin_z, torch.zeros(B, device=center.device, dtype=center.dtype)], dim=1),
        torch.stack([sin_z, cos_z, torch.zeros(B, device=center.device, dtype=center.dtype)], dim=1),
        torch.stack([torch.zeros(B, device=center.device, dtype=center.dtype), torch.zeros(B, device=center.device, dtype=center.dtype), torch.ones(B, device=center.device, dtype=center.dtype)], dim=1)
    ], dim=1)
    
    # Combined rotation matrix
    R = torch.matmul(torch.matmul(R_z, R_y), R_x)
    
    # Apply scaling
    S = torch.diag_embed(scales)
    RS = torch.matmul(R, S)
    
    # Construct the final projection matrix
    T = torch.zeros((B, 3, 4), device=center.device, dtype=center.dtype)
    T[:, :, :3] = RS
    T[:, :, 3] = center
    
    return T

