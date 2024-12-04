import torch

def sample_is_valid_for_homography(points1, points2):
    """
    Check if the sample is valid for homography estimation using the oriented constraint check.
    
    Args:
    points1 (torch.Tensor): Tensor of shape (B, 4, 2) representing points in the first image.
    points2 (torch.Tensor): Tensor of shape (B, 4, 2) representing points in the second image.
    
    Returns:
    torch.Tensor: A mask tensor of shape (B, 3, 3) representing the validity of the sample for each batch.
    """
    # Check if the shapes of points1 and points2 are equal
    if points1.shape != points2.shape:
        raise ValueError("The shapes of points1 and points2 must be equal.")
    
    B = points1.shape[0]
    
    # Convert points to homogeneous coordinates
    ones = torch.ones((B, 4, 1), dtype=points1.dtype, device=points1.device)
    points1_h = torch.cat([points1, ones], dim=-1)  # Shape: (B, 4, 3)
    points2_h = torch.cat([points2, ones], dim=-1)  # Shape: (B, 4, 3)
    
    # Function to compute the oriented constraint check
    def oriented_constraint(p1, p2):
        # Compute the cross product of the first three points
        cross_product = torch.cross(p1[:, :3], p1[:, 1:4], dim=-1)  # Shape: (B, 3, 3)
        # Compute the dot product with the fourth point
        dot_product = torch.einsum('bij,bj->bi', cross_product, p1[:, 3])
        # Check if the signs are consistent
        signs = torch.sign(dot_product)
        return torch.all(signs == signs[:, 0:1], dim=-1)
    
    # Apply the oriented constraint check to both sets of points
    valid1 = oriented_constraint(points1_h, points2_h)
    valid2 = oriented_constraint(points2_h, points1_h)
    
    # Combine the validity checks
    valid = valid1 & valid2
    
    # Create the mask tensor of shape (B, 3, 3)
    mask = valid.unsqueeze(-1).unsqueeze(-1).expand(-1, 3, 3)
    
    return mask

