import torch
import torch.nn.functional as F

def filter2d(input, kernel, border_type='constant', normalized=False, padding='same', behaviour='corr'):
    """
    Convolve a tensor with a 2d kernel.

    Args:
        input (torch.Tensor): the input tensor with shape of (B, C, H, W).
        kernel (torch.Tensor): the kernel to be convolved with the input tensor. 
                               The kernel shape must be (1, kH, kW) or (B, kH, kW).
        border_type (str): the padding mode to be applied before convolving. 
                           The expected modes are: 'constant', 'reflect', 'replicate' or 'circular'.
        normalized (bool): If True, kernel will be L1 normalized.
        padding (str): This defines the type of padding. 2 modes available 'same' or 'valid'.
        behaviour (str): defines the convolution mode -- correlation (default), using pytorch conv2d,
                         or true convolution (kernel is flipped). 2 modes available 'corr' or 'conv'.

    Returns:
        torch.Tensor: the convolved tensor of same size and numbers of channels as the input with shape (B, C, H, W).
    """
    B, C, H, W = input.shape
    if kernel.dim() == 3:
        kernel = kernel.unsqueeze(1)  # (B, 1, kH, kW)
    elif kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, kH, kW)
    else:
        raise ValueError("Kernel must be of shape (1, kH, kW) or (B, kH, kW)")

    if normalized:
        kernel = kernel / kernel.sum(dim=(-2, -1), keepdim=True)

    if behaviour == 'conv':
        kernel = torch.flip(kernel, dims=(-2, -1))

    if padding == 'same':
        pad_h = (kernel.shape[-2] - 1) // 2
        pad_w = (kernel.shape[-1] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, pad_h, pad_h), mode=border_type)
    elif padding != 'valid':
        raise ValueError("Padding must be 'same' or 'valid'")

    output = torch.zeros_like(input)
    for b in range(B):
        for c in range(C):
            output[b, c] = F.conv2d(input[b, c].unsqueeze(0).unsqueeze(0), kernel[b % kernel.shape[0]], padding=0).squeeze(0).squeeze(0)

    if padding == 'same':
        return output[:, :, pad_h:H+pad_h, pad_w:W+pad_w]
    return output

