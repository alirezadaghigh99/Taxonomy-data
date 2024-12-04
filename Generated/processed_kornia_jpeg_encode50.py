import torch
import torch.nn.functional as F
import numpy as np

def rgb_to_ycbcr(image_rgb):
    # Define the transformation matrix from RGB to YCbCr
    transform_matrix = torch.tensor([[0.299, 0.587, 0.114],
                                     [-0.168736, -0.331264, 0.5],
                                     [0.5, -0.418688, -0.081312]], dtype=image_rgb.dtype, device=image_rgb.device)
    shift = torch.tensor([0, 128, 128], dtype=image_rgb.dtype, device=image_rgb.device).view(1, 3, 1, 1)
    
    image_ycbcr = torch.tensordot(image_rgb.permute(0, 2, 3, 1), transform_matrix, dims=1).permute(0, 3, 1, 2) + shift
    return image_ycbcr

def block_split(image, block_size=8):
    B, C, H, W = image.shape
    image = image.view(B, C, H // block_size, block_size, W // block_size, block_size)
    image = image.permute(0, 1, 2, 4, 3, 5).contiguous()
    image = image.view(B, C, -1, block_size, block_size)
    return image

def dct_2d(image_block):
    return torch.fft.fft2(image_block, norm='ortho').real

def quantize(blocks, quant_table):
    return torch.round(blocks / quant_table)

def _jpeg_encode(image_rgb, jpeg_quality, quantization_table_y, quantization_table_c):
    B, C, H, W = image_rgb.shape
    assert C == 3, "Input images must have 3 channels (RGB)."
    
    # Convert RGB to YCbCr
    image_ycbcr = rgb_to_ycbcr(image_rgb)
    
    # Split into Y, Cb, Cr channels
    y_channel = image_ycbcr[:, 0:1, :, :]
    cb_channel = image_ycbcr[:, 1:2, :, :]
    cr_channel = image_ycbcr[:, 2:2, :, :]
    
    # Split into 8x8 blocks
    y_blocks = block_split(y_channel)
    cb_blocks = block_split(cb_channel)
    cr_blocks = block_split(cr_channel)
    
    # Apply DCT to each block
    y_dct = dct_2d(y_blocks)
    cb_dct = dct_2d(cb_blocks)
    cr_dct = dct_2d(cr_blocks)
    
    # Quantize the DCT coefficients
    y_quantized = quantize(y_dct, quantization_table_y)
    cb_quantized = quantize(cb_dct, quantization_table_c)
    cr_quantized = quantize(cr_dct, quantization_table_c)
    
    return y_quantized, cb_quantized, cr_quantized

