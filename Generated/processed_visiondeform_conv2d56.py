import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

def deform_conv2d(input, offset, weight, bias=None, stride=1, padding=0, dilation=1, mask=None):
    """
    Perform Deformable Convolution v2.

    Parameters:
    - input (Tensor): Input feature map of shape (N, C_in, H_in, W_in)
    - offset (Tensor): Offset tensor of shape (N, 2*kernel_h*kernel_w, H_out, W_out)
    - weight (Tensor): Convolution weight of shape (C_out, C_in, kernel_h, kernel_w)
    - bias (Tensor, optional): Bias tensor of shape (C_out,)
    - stride (int or tuple): Stride of the convolution
    - padding (int or tuple): Zero-padding added to both sides of the input
    - dilation (int or tuple): Spacing between kernel elements
    - mask (Tensor, optional): Modulation mask of shape (N, kernel_h*kernel_w, H_out, W_out)

    Returns:
    - output (Tensor): Output feature map of shape (N, C_out, H_out, W_out)
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    
    N, C_in, H_in, W_in = input.shape
    C_out, _, kernel_h, kernel_w = weight.shape
    
    H_out = (H_in + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    
    # Apply padding
    input_padded = F.pad(input, (padding[1], padding[1], padding[0], padding[0]))
    
    # Generate grid for sampling
    grid_y, grid_x = torch.meshgrid(torch.arange(H_out, device=input.device), torch.arange(W_out, device=input.device))
    grid_y = grid_y * stride[0]
    grid_x = grid_x * stride[1]
    grid = torch.stack((grid_x, grid_y), dim=-1).float()  # Shape: (H_out, W_out, 2)
    
    # Add offset
    offset = offset.permute(0, 2, 3, 1).contiguous()  # Shape: (N, H_out, W_out, 2*kernel_h*kernel_w)
    offset = offset.view(N, H_out, W_out, kernel_h, kernel_w, 2)
    grid = grid.unsqueeze(0).unsqueeze(3).unsqueeze(4)  # Shape: (1, H_out, W_out, 1, 1, 2)
    grid = grid + offset  # Shape: (N, H_out, W_out, kernel_h, kernel_w, 2)
    
    # Normalize grid to [-1, 1]
    grid[..., 0] = 2.0 * grid[..., 0] / (W_in - 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / (H_in - 1) - 1.0
    
    # Sample input at offset locations
    input_padded = input_padded.unsqueeze(1)  # Shape: (N, 1, C_in, H_in + 2*padding[0], W_in + 2*padding[1])
    grid = grid.view(N, H_out, W_out, kernel_h * kernel_w, 2)  # Shape: (N, H_out, W_out, kernel_h*kernel_w, 2)
    sampled_input = F.grid_sample(input_padded, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    sampled_input = sampled_input.view(N, C_in, H_out, W_out, kernel_h, kernel_w)  # Shape: (N, C_in, H_out, W_out, kernel_h, kernel_w)
    
    # Apply mask if provided
    if mask is not None:
        mask = mask.view(N, 1, H_out, W_out, kernel_h, kernel_w)
        sampled_input = sampled_input * mask
    
    # Perform convolution
    sampled_input = sampled_input.permute(0, 2, 3, 4, 5, 1).contiguous()  # Shape: (N, H_out, W_out, kernel_h, kernel_w, C_in)
    sampled_input = sampled_input.view(N, H_out, W_out, -1)  # Shape: (N, H_out, W_out, C_in*kernel_h*kernel_w)
    weight = weight.view(C_out, -1)  # Shape: (C_out, C_in*kernel_h*kernel_w)
    output = torch.einsum('nhwc,oc->nhwo', sampled_input, weight)  # Shape: (N, H_out, W_out, C_out)
    
    if bias is not None:
        output += bias.view(1, 1, 1, -1)
    
    return output.permute(0, 3, 1, 2).contiguous()  # Shape: (N, C_out, H_out, W_out)

