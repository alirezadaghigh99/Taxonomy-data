import torch

def safe_diff(u, t, order=1):
    # Validate input shapes
    if u.shape != t.shape:
        raise ValueError("u and t must have the same shape.")
    if u.ndim != 2 or u.shape[1] != 1:
        raise ValueError("u must have shape (n_samples, 1).")
    if t.ndim != 2 or t.shape[1] != 1:
        raise ValueError("t must have shape (n_samples, 1).")
    
    # Ensure that t requires gradient for differentiation
    t.requires_grad_(True)
    
    # Compute the derivative
    derivative = u
    for _ in range(order):
        # Compute the gradient of the current derivative with respect to t
        grad_outputs = torch.ones_like(derivative)
        derivative, = torch.autograd.grad(derivative, t, grad_outputs=grad_outputs, create_graph=True)
    
    return derivative

