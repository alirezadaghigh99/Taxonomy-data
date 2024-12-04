import torch
import torch.nn.functional as F

def yuv_to_rgb(yuv):
    """
    Convert a YUV444 image to RGB.
    Args:
        yuv (torch.Tensor): YUV image with shape (*, 3, H, W)
    Returns:
        torch.Tensor: RGB image with shape (*, 3, H, W)
    """
    # Define the conversion matrix from YUV to RGB
    matrix = torch.tensor([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ], dtype=yuv.dtype, device=yuv.device)
    
    # Separate Y, U, and V channels
    y = yuv[:, 0:1, :, :]
    u = yuv[:, 1:2, :, :]
    v = yuv[:, 2:3, :, :]
    
    # Concatenate Y, U, and V channels along the channel dimension
    yuv = torch.cat((y, u, v), dim=1)
    
    # Perform the matrix multiplication
    rgb = torch.tensordot(yuv.permute(0, 2, 3, 1), matrix, dims=([3], [1])).permute(0, 3, 1, 2)
    
    return rgb

def yuv420_to_rgb(imagey, imageuv):
    """
    Convert a YUV420 image to RGB.
    Args:
        imagey (torch.Tensor): Y (luma) image plane with shape (*, 1, H, W)
        imageuv (torch.Tensor): UV (chroma) image planes with shape (*, 2, H/2, W/2)
    Returns:
        torch.Tensor: RGB image with shape (*, 3, H, W)
    """
    # Ensure the input tensors are torch Tensors
    if not isinstance(imagey, torch.Tensor) or not isinstance(imageuv, torch.Tensor):
        raise TypeError("Input images must be torch Tensors")
    
    # Ensure the input tensors have the correct shapes
    if imagey.ndim != 4 or imageuv.ndim != 4:
        raise ValueError("Input images must have 4 dimensions")
    
    if imagey.shape[1] != 1 or imageuv.shape[1] != 2:
        raise ValueError("imagey must have shape (*, 1, H, W) and imageuv must have shape (*, 2, H/2, W/2)")
    
    # Get the dimensions
    _, _, H, W = imagey.shape
    _, _, H_uv, W_uv = imageuv.shape
    
    if H_uv * 2 != H or W_uv * 2 != W:
        raise ValueError("The dimensions of imageuv must be half of imagey")
    
    # Pad the input tensors to be evenly divisible by 2
    pad_h = (H % 2 != 0)
    pad_w = (W % 2 != 0)
    
    if pad_h or pad_w:
        imagey = F.pad(imagey, (0, pad_w, 0, pad_h), mode='reflect')
        imageuv = F.pad(imageuv, (0, pad_w // 2, 0, pad_h // 2), mode='reflect')
    
    # Upsample the UV planes to match the dimensions of the Y plane
    imageuv_upsampled = F.interpolate(imageuv, scale_factor=2, mode='bilinear', align_corners=False)
    
    # Concatenate the Y and upsampled UV planes to form a YUV444 image
    yuv444 = torch.cat((imagey, imageuv_upsampled), dim=1)
    
    # Convert the YUV444 image to RGB
    rgb = yuv_to_rgb(yuv444)
    
    return rgb

