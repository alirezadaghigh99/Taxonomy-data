import torch

def fundamental_from_projections(P1, P2):
    # Ensure the input tensors are of the correct shape
    assert P1.shape[-2:] == (3, 4), "P1 must have shape (*, 3, 4)"
    assert P2.shape[-2:] == (3, 4), "P2 must have shape (*, 3, 4)"
    assert P1.shape[:-2] == P2.shape[:-2], "P1 and P2 must have matching batch dimensions"
    
    # Convert inputs to float32 if they are not already in float32 or float64
    if P1.dtype not in [torch.float32, torch.float64]:
        P1 = P1.to(torch.float32)
    if P2.dtype not in [torch.float32, torch.float64]:
        P2 = P2.to(torch.float32)
    
    # Extract the batch dimensions
    batch_dims = P1.shape[:-2]
    
    # Initialize the fundamental matrix tensor
    F = torch.zeros(*batch_dims, 3, 3, dtype=P1.dtype, device=P1.device)
    
    # Compute the fundamental matrix using determinants of submatrices
    for i in range(3):
        for j in range(3):
            # Create a 4x4 matrix by combining P1 and P2
            M = torch.zeros(*batch_dims, 4, 4, dtype=P1.dtype, device=P1.device)
            M[..., :3, :] = P1
            M[..., 3, :] = P2[..., i, :]
            
            # Compute the determinant of the submatrix
            F[..., i, j] = torch.det(M)
    
    return F

