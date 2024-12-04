import torch
import torch.nn.functional as F

def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='none'):
    """
    Compute the focal loss between `inputs` and the ground truth `targets`.

    Args:
        inputs (torch.Tensor): Predictions for each example.
        targets (torch.Tensor): Binary classification labels for each element in `inputs`.
        alpha (float): Weighting factor for positive examples.
        gamma (float): Exponent factor to balance easy vs hard examples.
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', 'sum'.

    Returns:
        torch.Tensor: Loss tensor with the specified reduction applied.
    """
    # Ensure inputs and targets have the same shape
    assert inputs.shape == targets.shape, "Inputs and targets must have the same shape"

    # Compute the sigmoid of the inputs
    prob = torch.sigmoid(inputs)
    
    # Compute the binary cross entropy loss
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    
    # Compute the modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    modulating_factor = (1 - p_t) ** gamma
    
    # Compute the alpha factor
    alpha_factor = targets * alpha + (1 - targets) * (1 - alpha)
    
    # Compute the focal loss
    focal_loss = alpha_factor * modulating_factor * bce_loss
    
    # Apply the reduction option
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:  # 'none'
        return focal_loss

