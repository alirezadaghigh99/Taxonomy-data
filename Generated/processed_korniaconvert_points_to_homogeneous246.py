import torch

def convert_points_to_homogeneous(points):
    """
    Convert Euclidean points to homogeneous coordinates.
    
    Args:
        points (torch.Tensor): A tensor of shape (*, N, D) representing Euclidean points.
        
    Returns:
        torch.Tensor: A tensor of shape (*, N, D+1) representing the points in homogeneous coordinates.
        
    Raises:
        TypeError: If the input is not a tensor.
        ValueError: If the input tensor has fewer than two dimensions.
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError("Input must be a tensor.")
    
    if points.ndim < 2:
        raise ValueError("Input tensor must have at least two dimensions.")
    
    # Get the shape of the input tensor
    *batch_dims, N, D = points.shape
    
    # Create a tensor of ones with the same batch dimensions and N
    ones = torch.ones(*batch_dims, N, 1, dtype=points.dtype, device=points.device)
    
    # Concatenate the ones to the last dimension of the input points
    homogeneous_points = torch.cat([points, ones], dim=-1)
    
    return homogeneous_points

def _convert_affinematrix_to_homography_impl(matrix):
    """
    Convert an affine matrix to a homography matrix.
    
    Args:
        matrix (torch.Tensor): A tensor of shape (*, D, D+1) representing affine matrices.
        
    Returns:
        torch.Tensor: A tensor of shape (*, D+1, D+1) representing the homography matrices.
    """
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("Input must be a tensor.")
    
    if matrix.ndim < 2:
        raise ValueError("Input tensor must have at least two dimensions.")
    
    # Get the shape of the input tensor
    *batch_dims, D, D_plus_1 = matrix.shape
    
    if D_plus_1 != D + 1:
        raise ValueError("The last dimension of the input matrix must be D+1.")
    
    # Create an identity matrix of shape (D+1, D+1)
    identity = torch.eye(D + 1, dtype=matrix.dtype, device=matrix.device)
    
    # Create a tensor of zeros with the same batch dimensions and shape (D, 1)
    zeros = torch.zeros(*batch_dims, D, 1, dtype=matrix.dtype, device=matrix.device)
    
    # Concatenate the zeros to the last dimension of the input matrix
    top_part = torch.cat([matrix, zeros], dim=-1)
    
    # Concatenate the identity matrix to the bottom of the top_part
    homography_matrix = torch.cat([top_part, identity[-1:].expand(*batch_dims, -1, -1)], dim=-2)
    
    return homography_matrix

