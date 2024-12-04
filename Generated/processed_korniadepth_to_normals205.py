import torch
import torch.nn.functional as F

def depth_to_normals(depth, camera_matrix, normalize_points=True):
    # Validate input types
    if not isinstance(depth, torch.Tensor):
        raise TypeError("depth must be a Tensor")
    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError("camera_matrix must be a Tensor")
    
    # Validate input shapes
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("depth must have shape (B, 1, H, W)")
    if camera_matrix.ndim != 3 or camera_matrix.shape[1:] != (3, 3):
        raise ValueError("camera_matrix must have shape (B, 3, 3)")
    
    B, _, H, W = depth.shape
    
    # Create a meshgrid for pixel coordinates
    y, x = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device))
    y = y.float()
    x = x.float()
    
    # Expand the meshgrid to match the batch size
    y = y.unsqueeze(0).expand(B, -1, -1)
    x = x.unsqueeze(0).expand(B, -1, -1)
    
    # Get the camera intrinsics
    fx = camera_matrix[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
    fy = camera_matrix[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = camera_matrix[:, 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = camera_matrix[:, 1, 2].unsqueeze(-1).unsqueeze(-1)
    
    # Convert depth to 3D points
    z = depth.squeeze(1)
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    
    points = torch.stack((x, y, z), dim=1)
    
    if normalize_points:
        points = F.normalize(points, dim=1)
    
    # Compute gradients
    dzdx = F.pad(z[:, :, 1:] - z[:, :, :-1], (1, 0), mode='replicate')
    dzdy = F.pad(z[:, 1:, :] - z[:, :-1, :], (0, 0, 1, 0), mode='replicate')
    
    dx = F.pad(x[:, :, 1:] - x[:, :, :-1], (1, 0), mode='replicate')
    dy = F.pad(y[:, 1:, :] - y[:, :-1, :], (0, 0, 1, 0), mode='replicate')
    
    # Compute normals
    normals = torch.cross(torch.stack((dx, dy, dzdx), dim=1), torch.stack((dx, dy, dzdy), dim=1), dim=1)
    normals = F.normalize(normals, dim=1)
    
    return normals

