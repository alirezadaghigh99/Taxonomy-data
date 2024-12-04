import torch

def set_laf_orientation(LAF, angles_degrees):
    """
    Change the orientation of the Local Affine Frames (LAFs).

    Args:
        LAF (torch.Tensor): A tensor of shape (B, N, 2, 3) representing the LAFs.
        angles_degrees (torch.Tensor): A tensor of shape (B, N, 1) representing the angles in degrees.

    Returns:
        torch.Tensor: A tensor of shape (B, N, 2, 3) representing the LAFs oriented with the specified angles.
    """
    # Convert angles from degrees to radians
    angles_radians = torch.deg2rad(angles_degrees)

    # Compute the rotation matrices
    cos_angles = torch.cos(angles_radians)
    sin_angles = torch.sin(angles_radians)
    
    # Create rotation matrices of shape (B, N, 2, 2)
    rotation_matrices = torch.zeros(LAF.shape[0], LAF.shape[1], 2, 2, device=LAF.device)
    rotation_matrices[:, :, 0, 0] = cos_angles.squeeze(-1)
    rotation_matrices[:, :, 0, 1] = -sin_angles.squeeze(-1)
    rotation_matrices[:, :, 1, 0] = sin_angles.squeeze(-1)
    rotation_matrices[:, :, 1, 1] = cos_angles.squeeze(-1)

    # Apply the rotation to the LAFs
    LAF_rotated = torch.zeros_like(LAF)
    LAF_rotated[:, :, :2, :2] = torch.matmul(rotation_matrices, LAF[:, :, :2, :2])
    LAF_rotated[:, :, :2, 2] = LAF[:, :, :2, 2]

    return LAF_rotated

