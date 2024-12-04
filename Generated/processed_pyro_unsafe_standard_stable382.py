import torch
import math

def _unsafe_standard_stable(alpha, beta, V, W, coords):
    # Ensure V and W have the same shape
    assert V.shape == W.shape, "V and W must have the same shape"
    
    # Constants for numerical stability
    eps = 1e-10
    
    # Check if alpha is close to 1
    if abs(alpha - 1.0) < eps:
        raise ValueError("Alpha is too close to 1, which is not supported.")
    
    # Precompute some constants
    b = math.atan(beta * math.tan(math.pi * alpha / 2)) / alpha
    S = (1 + beta**2 * math.tan(math.pi * alpha / 2)**2)**(1 / (2 * alpha))
    
    # Generate the alpha-stable random variables
    if coords == "S0":
        theta = math.pi * (V - 0.5)
        W = -torch.log(W)
        Z = S * (torch.sin(alpha * theta + b) / (torch.cos(theta)**(1 / alpha))) * \
            ((torch.cos(theta - alpha * theta - b) / W)**((1 - alpha) / alpha))
    elif coords == "S":
        theta = math.pi * (V - 0.5)
        W = -torch.log(W)
        Z = S * (torch.sin(alpha * theta + b) / (torch.cos(theta)**(1 / alpha))) * \
            ((torch.cos(theta - alpha * theta - b) / W)**((1 - alpha) / alpha))
    else:
        raise ValueError(f"Unknown coords: {coords}")
    
    # Replace NaN values with zeros
    Z = torch.where(torch.isnan(Z), torch.zeros_like(Z), Z)
    
    return Z

