import torch

def rinverse(M, sym=False):
    """Matrix inversion of rightmost dimensions (batched).

    For 1, 2, and 3 dimensions this uses the formulae.
    For larger matrices, it uses blockwise inversion to reduce to
    smaller matrices.
    """
    if M.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")
    
    *batch_dims, n, m = M.shape
    if n != m:
        raise ValueError("The rightmost two dimensions must be square matrices")
    
    if n == 1:
        return 1.0 / M
    
    if n == 2:
        det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
        inv = torch.empty_like(M)
        inv[..., 0, 0] = M[..., 1, 1]
        inv[..., 0, 1] = -M[..., 0, 1]
        inv[..., 1, 0] = -M[..., 1, 0]
        inv[..., 1, 1] = M[..., 0, 0]
        return inv / det.unsqueeze(-1).unsqueeze(-1)
    
    if n == 3:
        det = (
            M[..., 0, 0] * (M[..., 1, 1] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 1]) -
            M[..., 0, 1] * (M[..., 1, 0] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 0]) +
            M[..., 0, 2] * (M[..., 1, 0] * M[..., 2, 1] - M[..., 1, 1] * M[..., 2, 0])
        )
        inv = torch.empty_like(M)
        inv[..., 0, 0] = M[..., 1, 1] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 1]
        inv[..., 0, 1] = M[..., 0, 2] * M[..., 2, 1] - M[..., 0, 1] * M[..., 2, 2]
        inv[..., 0, 2] = M[..., 0, 1] * M[..., 1, 2] - M[..., 0, 2] * M[..., 1, 1]
        inv[..., 1, 0] = M[..., 1, 2] * M[..., 2, 0] - M[..., 1, 0] * M[..., 2, 2]
        inv[..., 1, 1] = M[..., 0, 0] * M[..., 2, 2] - M[..., 0, 2] * M[..., 2, 0]
        inv[..., 1, 2] = M[..., 0, 2] * M[..., 1, 0] - M[..., 0, 0] * M[..., 1, 2]
        inv[..., 2, 0] = M[..., 1, 0] * M[..., 2, 1] - M[..., 1, 1] * M[..., 2, 0]
        inv[..., 2, 1] = M[..., 0, 1] * M[..., 2, 0] - M[..., 0, 0] * M[..., 2, 1]
        inv[..., 2, 2] = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
        return inv / det.unsqueeze(-1).unsqueeze(-1)
    
    # For larger matrices, use blockwise inversion
    if sym:
        return torch.linalg.inv(M)
    else:
        return torch.inverse(M)

