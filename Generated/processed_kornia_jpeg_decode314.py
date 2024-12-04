import torch
import torch.nn.functional as F
import numpy as np

def _jpeg_decode(input_y, input_cb, input_cr, jpeg_quality, H, W, quantization_table_y, quantization_table_c):
    """
    Perform JPEG decoding.

    Args:
        input_y (Tensor): Compressed Y component of the shape (B, N, 8, 8).
        input_cb (Tensor): Compressed Cb component of the shape (B, N, 8, 8).
        input_cr (Tensor): Compressed Cr component of the shape (B, N, 8, 8).
        jpeg_quality (Tensor): Compression strength of the shape (B).
        H (int): Original image height.
        W (int): Original image width.
        quantization_table_y (Tensor): Quantization table for Y channel.
        quantization_table_c (Tensor): Quantization table for C channels.

    Returns:
        rgb_decoded (Tensor): Decompressed RGB image of the shape (B, 3, H, W).
    """
    B = input_y.size(0)
    
    # Dequantization
    dequantized_y = input_y * quantization_table_y.unsqueeze(0).unsqueeze(0)
    dequantized_cb = input_cb * quantization_table_c.unsqueeze(0).unsqueeze(0)
    dequantized_cr = input_cr * quantization_table_c.unsqueeze(0).unsqueeze(0)
    
    # Inverse DCT
    def idct_2d(block):
        return torch.idct(block, norm='ortho', dim=-1).idct(norm='ortho', dim=-2)
    
    y_blocks = dequantized_y.view(B, -1, 8, 8)
    cb_blocks = dequantized_cb.view(B, -1, 8, 8)
    cr_blocks = dequantized_cr.view(B, -1, 8, 8)
    
    y_blocks = idct_2d(y_blocks)
    cb_blocks = idct_2d(cb_blocks)
    cr_blocks = idct_2d(cr_blocks)
    
    # Reconstruct the image from blocks
    def blocks_to_image(blocks, height, width):
        B, num_blocks, block_size, _ = blocks.size()
        blocks_per_row = width // block_size
        blocks_per_col = height // block_size
        image = blocks.view(B, blocks_per_col, blocks_per_row, block_size, block_size)
        image = image.permute(0, 1, 3, 2, 4).contiguous()
        image = image.view(B, height, width)
        return image
    
    y_image = blocks_to_image(y_blocks, H, W)
    cb_image = blocks_to_image(cb_blocks, H // 2, W // 2)
    cr_image = blocks_to_image(cr_blocks, H // 2, W // 2)
    
    # Upsample Cb and Cr to match Y dimensions
    cb_image = F.interpolate(cb_image.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
    cr_image = F.interpolate(cr_image.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
    
    # Convert YCbCr to RGB
    def ycbcr_to_rgb(y, cb, cr):
        r = y + 1.402 * (cr - 128)
        g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
        b = y + 1.772 * (cb - 128)
        return torch.stack((r, g, b), dim=1)
    
    rgb_image = ycbcr_to_rgb(y_image, cb_image, cr_image)
    
    # Clip values to [0, 255] and convert to uint8
    rgb_image = torch.clamp(rgb_image, 0, 255).byte()
    
    return rgb_image

