import torch

def _get_perspective_coeffs(startpoints, endpoints):
    if len(startpoints) != 4 or len(endpoints) != 4:
        raise ValueError("Both startpoints and endpoints must contain exactly four coordinate pairs.")
    
    # Convert points to tensors
    startpoints = torch.tensor(startpoints, dtype=torch.float64)
    endpoints = torch.tensor(endpoints, dtype=torch.float64)
    
    # Prepare the matrix A and vector B for the equation A * coeffs = B
    A = []
    B = []
    
    for (x, y), (u, v) in zip(startpoints, endpoints):
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
        B.append(u)
        B.append(v)
    
    A = torch.tensor(A, dtype=torch.float64)
    B = torch.tensor(B, dtype=torch.float64)
    
    # Solve the system using least squares
    coeffs, _ = torch.lstsq(B, A)
    
    # The result from torch.lstsq includes extra rows, we only need the first 8 coefficients
    coeffs = coeffs[:8].flatten()
    
    # Convert to single precision before returning
    coeffs = coeffs.to(dtype=torch.float32)
    
    return coeffs.tolist()

