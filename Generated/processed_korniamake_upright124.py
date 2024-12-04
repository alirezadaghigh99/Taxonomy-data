import torch

def KORNIA_CHECK_LAF(laf):
    if laf.ndimension() != 4 or laf.size(2) != 2 or laf.size(3) != 3:
        raise ValueError("Input LAF should be of shape (B, N, 2, 3)")

def make_upright(laf, eps=1e-6):
    # Validate the input affine matrix
    KORNIA_CHECK_LAF(laf)
    
    # Extract the 2x2 affine part of the matrix
    A = laf[:, :, :2, :2]
    
    # Calculate the determinant of the affine matrix
    det = torch.det(A)
    
    # Perform SVD on the 2x2 matrix
    U, S, V = torch.svd(A)
    
    # Set the rotation matrix to an identity matrix
    identity_matrix = torch.eye(2, device=laf.device).unsqueeze(0).unsqueeze(0)
    identity_matrix = identity_matrix.expand_as(A)
    
    # Scale the rectified affine matrix using the original determinant
    scale_factor = torch.sqrt(det).unsqueeze(-1).unsqueeze(-1)
    rectified_A = identity_matrix * scale_factor
    
    # Replace the original 2x2 affine part with the rectified one
    rectified_laf = laf.clone()
    rectified_laf[:, :, :2, :2] = rectified_A
    
    return rectified_laf

