import torch

def quaternion_exp_to_log(quaternion, eps):
    # Check if the input is a tensor
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    # Check if the input has the correct shape (*, 4)
    if quaternion.shape[-1] != 4:
        raise ValueError("Input tensor must have shape (*, 4)")
    
    # Unpack the quaternion into scalar (w) and vector (x, y, z)
    w = quaternion[..., 0]
    xyz = quaternion[..., 1:]
    
    # Compute the norm of the vector part
    norm_xyz = torch.norm(xyz, dim=-1)
    
    # Compute the angle theta
    theta = torch.atan2(norm_xyz, w)
    
    # Avoid division by zero by clamping norm_xyz
    norm_xyz = torch.clamp(norm_xyz, min=eps)
    
    # Compute the log map
    log_map = theta.unsqueeze(-1) * (xyz / norm_xyz.unsqueeze(-1))
    
    return log_map

