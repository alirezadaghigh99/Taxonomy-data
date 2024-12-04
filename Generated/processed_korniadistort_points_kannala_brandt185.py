import torch

def distort_points_kannala_brandt(projected_points_in_camera_z1_plane, params):
    # Extract intrinsic parameters and distortion coefficients
    fx, fy, cx, cy, k1, k2, k3, k4 = params[..., 0], params[..., 1], params[..., 2], params[..., 3], params[..., 4], params[..., 5], params[..., 6], params[..., 7]
    
    # Extract x and y coordinates from the input points
    x = projected_points_in_camera_z1_plane[..., 0]
    y = projected_points_in_camera_z1_plane[..., 1]
    
    # Compute the radial distance from the center
    r = torch.sqrt(x**2 + y**2)
    
    # Compute the distorted radius using the Kannala-Brandt model
    theta = torch.atan(r)
    theta2 = theta**2
    theta4 = theta2**2
    theta6 = theta4 * theta2
    theta8 = theta4**2
    
    theta_d = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9
    
    # Avoid division by zero
    r = torch.where(r == 0, torch.tensor(1e-6, dtype=r.dtype, device=r.device), r)
    
    # Compute the scaling factor
    scale = theta_d / r
    
    # Apply the distortion
    x_distorted = x * scale
    y_distorted = y * scale
    
    # Map back to image coordinates
    u = fx * x_distorted + cx
    v = fy * y_distorted + cy
    
    # Stack the results to get the final distorted points
    distorted_points = torch.stack([u, v], dim=-1)
    
    return distorted_points

