import torch

def aepe(input, target, reduction='mean'):
    """
    Calculate the Average Endpoint Error (AEPE) between two flow maps.

    Args:
        input (torch.Tensor): the input flow map with shape (*, 2).
        target (torch.Tensor): the target flow map with shape (*, 2).
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                         'none': no reduction will be applied,
                         'mean': the sum of the output will be divided by the number of elements,
                         'sum': the output will be summed.

    Returns:
        torch.Tensor: the computed AEPE as a scalar.
    """
    if input.shape != target.shape:
        raise ValueError("Input and target must have the same shape")

    # Calculate the squared differences
    diff = input - target
    squared_diff = diff ** 2

    # Sum the squared differences along the last dimension (2D vectors)
    sum_squared_diff = torch.sum(squared_diff, dim=-1)

    # Calculate the square root of the sum of squared differences
    endpoint_error = torch.sqrt(sum_squared_diff)

    if reduction == 'none':
        return endpoint_error
    elif reduction == 'mean':
        return torch.mean(endpoint_error)
    elif reduction == 'sum':
        return torch.sum(endpoint_error)
    else:
        raise ValueError("Reduction must be one of 'none', 'mean', or 'sum'")

