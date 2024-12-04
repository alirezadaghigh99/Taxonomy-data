import torch

def rotate_laf(LAF, angles_degrees):
    # Validate input shapes
    if LAF.ndimension() != 4 or LAF.shape[2:] != (2, 3):
        raise ValueError("LAF must be of shape (B, N, 2, 3)")
    if angles_degrees.ndimension() != 3 or angles_degrees.shape[2] != 1:
        raise ValueError("angles_degrees must be of shape (B, N, 1)")
    if LAF.shape[:2] != angles_degrees.shape[:2]:
        raise ValueError("LAF and angles_degrees must have matching batch and point dimensions")

    # Convert angles from degrees to radians
    angles_radians = torch.deg2rad(angles_degrees)

    # Create rotation matrices
    cos_angles = torch.cos(angles_radians)
    sin_angles = torch.sin(angles_radians)
    rotation_matrices = torch.zeros_like(LAF)
    rotation_matrices[:, :, 0, 0] = cos_angles.squeeze(-1)
    rotation_matrices[:, :, 0, 1] = -sin_angles.squeeze(-1)
    rotation_matrices[:, :, 1, 0] = sin_angles.squeeze(-1)
    rotation_matrices[:, :, 1, 1] = cos_angles.squeeze(-1)
    rotation_matrices[:, :, :, 2] = LAF[:, :, :, 2]  # Copy the translation part

    # Apply the rotation to the LAF
    rotated_LAF = torch.zeros_like(LAF)
    rotated_LAF[:, :, :, :2] = torch.matmul(rotation_matrices[:, :, :, :2], LAF[:, :, :, :2])
    rotated_LAF[:, :, :, 2] = LAF[:, :, :, 2]  # Copy the translation part

    return rotated_LAF

