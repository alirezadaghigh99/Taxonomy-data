import torch

def cross_product_matrix(x: torch.Tensor) -> torch.Tensor:
    if not x.shape[-1] == 3:
        raise AssertionError(x.shape)
    
    # Extract the components of the vector
    a = x[..., 0]
    b = x[..., 1]
    c = x[..., 2]
    
    # Create the skew-symmetric matrix
    zero = torch.zeros_like(a)
    cross_prod_mat = torch.stack([
        torch.stack([zero, -c, b], dim=-1),
        torch.stack([c, zero, -a], dim=-1),
        torch.stack([-b, a, zero], dim=-1)
    ], dim=-2)
    
    return cross_prod_mat

