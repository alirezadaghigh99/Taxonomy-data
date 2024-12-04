import torch

def axis_angle_to_rotation_matrix(axis_angle):
    """
    Convert 3D vector of axis-angle rotation to 3x3 rotation matrix.

    Args:
        axis_angle: tensor of 3D vector of axis-angle rotations in radians with shape (N, 3).

    Returns:
        tensor of rotation matrices of shape (N, 3, 3).
    """
    # Ensure the input is a tensor
    if not isinstance(axis_angle, torch.Tensor):
        axis_angle = torch.tensor(axis_angle, dtype=torch.float32)
    
    # Get the batch size
    N = axis_angle.shape[0]
    
    # Compute the angle (magnitude of the axis-angle vector)
    angles = torch.norm(axis_angle, dim=1, keepdim=True)
    
    # Avoid division by zero by setting zero angles to a small value
    angles = torch.where(angles == 0, torch.tensor(1e-8, dtype=angles.dtype), angles)
    
    # Normalize the axis vectors
    axis = axis_angle / angles
    
    # Compute the components of the Rodrigues' rotation formula
    cos_angles = torch.cos(angles).unsqueeze(-1)
    sin_angles = torch.sin(angles).unsqueeze(-1)
    one_minus_cos = 1 - cos_angles
    
    # Extract the components of the normalized axis vectors
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    
    # Compute the rotation matrices
    rotation_matrices = torch.zeros((N, 3, 3), dtype=axis_angle.dtype)
    rotation_matrices[:, 0, 0] = cos_angles.squeeze() + x * x * one_minus_cos.squeeze()
    rotation_matrices[:, 0, 1] = x * y * one_minus_cos.squeeze() - z * sin_angles.squeeze()
    rotation_matrices[:, 0, 2] = x * z * one_minus_cos.squeeze() + y * sin_angles.squeeze()
    rotation_matrices[:, 1, 0] = y * x * one_minus_cos.squeeze() + z * sin_angles.squeeze()
    rotation_matrices[:, 1, 1] = cos_angles.squeeze() + y * y * one_minus_cos.squeeze()
    rotation_matrices[:, 1, 2] = y * z * one_minus_cos.squeeze() - x * sin_angles.squeeze()
    rotation_matrices[:, 2, 0] = z * x * one_minus_cos.squeeze() - y * sin_angles.squeeze()
    rotation_matrices[:, 2, 1] = z * y * one_minus_cos.squeeze() + x * sin_angles.squeeze()
    rotation_matrices[:, 2, 2] = cos_angles.squeeze() + z * z * one_minus_cos.squeeze()
    
    return rotation_matrices

