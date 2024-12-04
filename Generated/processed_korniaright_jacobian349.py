import torch

def vector_to_skew_symmetric_matrix(vec):
    """
    Converts a vector of shape (B, 3) to its corresponding skew-symmetric matrix.
    
    Args:
        vec (torch.Tensor): A tensor of shape (B, 3).
        
    Returns:
        torch.Tensor: A tensor of shape (B, 3, 3) representing the skew-symmetric matrices.
    """
    B = vec.shape[0]
    skew_sym = torch.zeros((B, 3, 3), dtype=vec.dtype, device=vec.device)
    skew_sym[:, 0, 1] = -vec[:, 2]
    skew_sym[:, 0, 2] = vec[:, 1]
    skew_sym[:, 1, 0] = vec[:, 2]
    skew_sym[:, 1, 2] = -vec[:, 0]
    skew_sym[:, 2, 0] = -vec[:, 1]
    skew_sym[:, 2, 1] = vec[:, 0]
    return skew_sym

def right_jacobian(vec):
    """
    Computes the right Jacobian of SO(3) for a given tensor of shape (B, 3).
    
    Args:
        vec (torch.Tensor): A tensor of shape (B, 3).
        
    Returns:
        torch.Tensor: A tensor of shape (B, 3, 3) representing the right Jacobian matrices.
    """
    B = vec.shape[0]
    norm_vec = torch.norm(vec, dim=1, keepdim=True)
    norm_vec = norm_vec + 1e-8  # To avoid division by zero
    
    skew_sym = vector_to_skew_symmetric_matrix(vec)
    
    eye = torch.eye(3, dtype=vec.dtype, device=vec.device).unsqueeze(0).repeat(B, 1, 1)
    
    term1 = (1 - torch.cos(norm_vec)) / (norm_vec ** 2)
    term2 = (norm_vec - torch.sin(norm_vec)) / (norm_vec ** 3)
    
    term1 = term1.view(B, 1, 1)
    term2 = term2.view(B, 1, 1)
    
    right_jacobian = eye + term1 * skew_sym + term2 * torch.bmm(skew_sym, skew_sym)
    
    return right_jacobian

