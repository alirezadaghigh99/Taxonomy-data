import torch
import torch.nn.functional as F

def spatial_gradient(input_image, mode='sobel', order=1, normalized=False):
    """
    Computes the first-order image derivative in both x and y directions.

    Parameters:
    - input_image (torch.Tensor): Input image tensor with shape (B, C, H, W).
    - mode (str): Derivatives modality, either 'sobel' or 'diff'.
    - order (int): Order of the derivatives.
    - normalized (bool): Whether to normalize the output.

    Returns:
    - torch.Tensor: Derivatives of the input feature map with shape (B, C, 2, H, W).
    """
    if mode not in ['sobel', 'diff']:
        raise ValueError("Mode must be either 'sobel' or 'diff'")
    if order != 1:
        raise ValueError("Currently only first-order derivatives are supported")

    B, C, H, W = input_image.shape

    if mode == 'sobel':
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=input_image.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=input_image.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(input_image.view(B * C, 1, H, W), sobel_x, padding=1).view(B, C, H, W)
        grad_y = F.conv2d(input_image.view(B * C, 1, H, W), sobel_y, padding=1).view(B, C, H, W)
    
    elif mode == 'diff':
        grad_x = input_image[:, :, :, 1:] - input_image[:, :, :, :-1]
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        
        grad_y = input_image[:, :, 1:, :] - input_image[:, :, :-1, :]
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')

    if normalized:
        grad_x = grad_x / torch.max(torch.abs(grad_x))
        grad_y = grad_y / torch.max(torch.abs(grad_y))

    gradients = torch.stack((grad_x, grad_y), dim=2)
    return gradients

