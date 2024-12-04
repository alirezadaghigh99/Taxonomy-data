import torch

def diff(u, t, order=1, shape_check=True):
    """
    Calculate the derivative of u with respect to t.

    Parameters:
    - u (torch.Tensor): The tensor representing the function values.
    - t (torch.Tensor): The tensor representing the variable with respect to which the derivative is taken.
    - order (int): The order of the derivative. Default is 1.
    - shape_check (bool): Whether to perform shape checking. Default is True.

    Returns:
    - torch.Tensor: The derivative of u with respect to t.
    """
    if shape_check:
        if u.shape != t.shape:
            raise ValueError("The shapes of u and t must be the same for differentiation.")

    # Ensure t requires gradient for autograd
    t = t.clone().detach().requires_grad_(True)

    # Compute the derivative
    derivative = u
    for _ in range(order):
        # Compute the gradient of the current derivative
        grad_outputs = torch.ones_like(derivative)
        derivative, = torch.autograd.grad(derivative, t, grad_outputs=grad_outputs, create_graph=True)

    return derivative

