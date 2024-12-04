import torch

def undistort_points_kannala_brandt(distorted_points_in_camera, params, max_iterations=10, tolerance=1e-6):
    """
    Undistorts points from the camera frame into the canonical z=1 plane using the Kannala-Brandt model.
    
    Args:
        distorted_points_in_camera (Tensor): Points to undistort with shape (..., 2).
        params (Tensor): Parameters of the Kannala-Brandt distortion model with shape (..., 8).
        max_iterations (int): Maximum number of iterations for the Gauss-Newton optimization.
        tolerance (float): Tolerance for the stopping criterion.
    
    Returns:
        Tensor: Undistorted points with shape (..., 2).
    """
    def kannala_brandt_model(x, y, params):
        k1, k2, k3, k4, g1, g2, g3, g4 = params
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan(r)
        theta2 = theta**2
        theta4 = theta2**2
        theta6 = theta4 * theta2
        theta8 = theta4**2
        theta_d = theta * (1 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8)
        scale = torch.where(r > 0, theta_d / r, torch.ones_like(r))
        return scale * x, scale * y

    def jacobian(x, y, params):
        k1, k2, k3, k4, g1, g2, g3, g4 = params
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan(r)
        theta2 = theta**2
        theta4 = theta2**2
        theta6 = theta4 * theta2
        theta8 = theta4**2
        theta_d = theta * (1 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8)
        dtheta_dr = 1 / (1 + r**2)
        dtheta_d_dtheta = 1 + 3*k1*theta2 + 5*k2*theta4 + 7*k3*theta6 + 9*k4*theta8
        dtheta_d_dr = dtheta_d_dtheta * dtheta_dr
        scale = torch.where(r > 0, theta_d / r, torch.ones_like(r))
        dscale_dr = torch.where(r > 0, (dtheta_d_dr * r - theta_d) / r**2, torch.zeros_like(r))
        J = torch.zeros(x.shape + (2, 2), dtype=x.dtype, device=x.device)
        J[..., 0, 0] = scale + x**2 * dscale_dr / r
        J[..., 0, 1] = x * y * dscale_dr / r
        J[..., 1, 0] = x * y * dscale_dr / r
        J[..., 1, 1] = scale + y**2 * dscale_dr / r
        return J

    undistorted_points = distorted_points_in_camera.clone()
    for _ in range(max_iterations):
        x, y = undistorted_points[..., 0], undistorted_points[..., 1]
        u, v = kannala_brandt_model(x, y, params)
        residuals = torch.stack([u, v], dim=-1) - distorted_points_in_camera
        J = jacobian(x, y, params)
        J_inv = torch.linalg.pinv(J)
        delta = torch.einsum('...ij,...j->...i', J_inv, residuals)
        undistorted_points -= delta
        if torch.max(torch.abs(delta)) < tolerance:
            break

    return undistorted_points

