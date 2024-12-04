import torch

def convert_points_from_homogeneous(points, eps=1e-10):
    # Check if the input is a tensor
    if not isinstance(points, torch.Tensor):
        raise TypeError("Input must be a tensor.")
    
    # Check if the input tensor has at least two dimensions
    if points.dim() < 2:
        raise ValueError("Input tensor must have at least two dimensions.")
    
    # Extract the last dimension (homogeneous coordinate)
    homogeneous_coord = points[..., -1:]
    
    # Avoid division by zero by adding eps to the homogeneous coordinate
    homogeneous_coord = homogeneous_coord + eps
    
    # Divide the other coordinates by the homogeneous coordinate
    euclidean_points = points[..., :-1] / homogeneous_coord
    
    return euclidean_points

