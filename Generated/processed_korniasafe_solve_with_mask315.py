import torch
import warnings

def safe_solve_with_mask(B, A):
    """
    Solves the system of linear equations AX = B while handling singular matrices.
    
    Args:
    - B (Tensor): The right-hand side tensor in the equation AX = B.
    - A (Tensor): The matrix tensor that will be solved against B.
    
    Returns:
    - X (Tensor): The solution tensor that satisfies AX = B, or a closest approximation if the matrix is near-singular.
    - A_LU (Tensor): The LU decomposition of matrix A, which is useful for numerical stability.
    - valid_mask (Tensor): A boolean tensor indicating which rows of the batch were solved successfully.
    """
    
    # Check if B is a tensor
    assert isinstance(B, torch.Tensor), "B must be a tensor"
    
    # Ensure B is of type float32 or float64
    if B.dtype not in [torch.float32, torch.float64]:
        B = B.to(torch.float32)
    
    # Check PyTorch version
    pytorch_version = torch.__version__.split('.')
    major_version = int(pytorch_version[0])
    minor_version = int(pytorch_version[1])
    
    if major_version < 1 or (major_version == 1 and minor_version < 10):
        warnings.warn("PyTorch version is less than 1.10. Falling back to _torch_solve_cast method. Validity mask might not be correct.")
        return _torch_solve_cast(B, A)
    
    try:
        # Perform LU decomposition
        A_LU, pivots = torch.lu(A)
        
        # Solve the system using LU decomposition
        X = torch.lu_solve(B, A_LU, pivots)
        
        # Check for singular matrices
        valid_mask = torch.isfinite(X).all(dim=-1)
        
        return X, A_LU, valid_mask
    
    except RuntimeError as e:
        # Handle singular matrix case
        if 'singular' in str(e):
            valid_mask = torch.zeros(B.shape[0], dtype=torch.bool)
            X = torch.zeros_like(B)
            A_LU = torch.zeros_like(A)
            return X, A_LU, valid_mask
        else:
            raise e

def _torch_solve_cast(B, A):
    """
    Fallback method for solving AX = B for PyTorch versions < 1.10.
    
    Args:
    - B (Tensor): The right-hand side tensor in the equation AX = B.
    - A (Tensor): The matrix tensor that will be solved against B.
    
    Returns:
    - X (Tensor): The solution tensor that satisfies AX = B.
    - A_LU (Tensor): The LU decomposition of matrix A.
    - valid_mask (Tensor): A boolean tensor indicating which rows of the batch were solved successfully.
    """
    try:
        # Perform LU decomposition
        A_LU, pivots = torch.lu(A)
        
        # Solve the system using LU decomposition
        X = torch.lu_solve(B, A_LU, pivots)
        
        # Check for singular matrices
        valid_mask = torch.isfinite(X).all(dim=-1)
        
        return X, A_LU, valid_mask
    
    except RuntimeError as e:
        # Handle singular matrix case
        if 'singular' in str(e):
            valid_mask = torch.zeros(B.shape[0], dtype=torch.bool)
            X = torch.zeros_like(B)
            A_LU = torch.zeros_like(A)
            return X, A_LU, valid_mask
        else:
            raise e