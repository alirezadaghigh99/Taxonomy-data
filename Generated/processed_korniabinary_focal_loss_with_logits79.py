import torch
import torch.nn.functional as F

def binary_focal_loss_with_logits(pred, target, alpha=0.25, gamma=2.0, reduction='mean', pos_weight=None, weight=None):
    """
    Compute the Binary Focal Loss with logits.

    Args:
        pred (torch.Tensor): Logits tensor with shape (N, C, *) where C = number of classes.
        target (torch.Tensor): Labels tensor with the same shape as pred (N, C, *) where each value is between 0 and 1.
        alpha (float): Weighting factor alpha in [0, 1].
        gamma (float): Focusing parameter gamma >= 0.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        pos_weight (torch.Tensor, optional): A weight of positive examples with shape (num_of_classes,).
        weight (torch.Tensor, optional): Weights for classes with shape (num_of_classes,).

    Returns:
        torch.Tensor: The computed loss.
    """
    # Convert logits to probabilities
    prob = torch.sigmoid(pred)
    
    # Compute the binary cross-entropy loss
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none', pos_weight=pos_weight)
    
    # Compute the focal loss
    p_t = prob * target + (1 - prob) * (1 - target)
    focal_loss = alpha * (1 - p_t) ** gamma * bce_loss
    
    # Apply class weights if provided
    if weight is not None:
        focal_loss = focal_loss * weight
    
    # Apply reduction method
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss

