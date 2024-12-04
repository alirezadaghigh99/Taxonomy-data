import torch

def quaternion_to_rotation_matrix(quaternion):
    # Normalize the quaternion
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    
    # Unpack the quaternion components
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    
    # Compute the rotation matrix elements
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    
    # Create the rotation matrix
    rotation_matrix = torch.empty((*quaternion.shape[:-1], 3, 3), dtype=quaternion.dtype, device=quaternion.device)
    rotation_matrix[..., 0, 0] = 1 - 2 * (yy + zz)
    rotation_matrix[..., 0, 1] = 2 * (xy - wz)
    rotation_matrix[..., 0, 2] = 2 * (xz + wy)
    rotation_matrix[..., 1, 0] = 2 * (xy + wz)
    rotation_matrix[..., 1, 1] = 1 - 2 * (xx + zz)
    rotation_matrix[..., 1, 2] = 2 * (yz - wx)
    rotation_matrix[..., 2, 0] = 2 * (xz - wy)
    rotation_matrix[..., 2, 1] = 2 * (yz + wx)
    rotation_matrix[..., 2, 2] = 1 - 2 * (xx + yy)
    
    return rotation_matrix

