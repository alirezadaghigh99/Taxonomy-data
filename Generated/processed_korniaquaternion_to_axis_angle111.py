import torch

def quaternion_to_axis_angle(quaternion):
    # Check if the input is a tensor
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input must be a tensor.")
    
    # Check if the input tensor has the correct shape
    if quaternion.dim() == 1 and quaternion.shape[0] == 4:
        quaternion = quaternion.unsqueeze(0)
    elif quaternion.dim() == 2 and quaternion.shape[1] == 4:
        pass
    else:
        raise ValueError("Input tensor must have shape Nx4 or 4.")
    
    # Unpack the quaternion components
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    
    # Compute the angle of rotation
    angle = 2 * torch.acos(w)
    
    # Compute the axis of rotation
    sin_half_angle = torch.sqrt(1 - w**2)
    axis = torch.stack((x, y, z), dim=1)
    
    # Normalize the axis
    axis = torch.where(sin_half_angle.unsqueeze(1) > 0, axis / sin_half_angle.unsqueeze(1), axis)
    
    # Handle the case when sin_half_angle is zero (angle is 0 or 2*pi)
    axis = torch.where(sin_half_angle.unsqueeze(1) == 0, torch.zeros_like(axis), axis)
    
    # Combine the axis and angle into the final output
    axis_angle = torch.cat((axis, angle.unsqueeze(1)), dim=1)
    
    return axis_angle.squeeze()

