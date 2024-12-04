import torch
import torch.nn.functional as F

def extract_patches_from_pyramid(img, laf, PS=32, normalize_lafs=False):
    """
    Extract image patches from a pyramid of images based on input Local Affine Frames (LAFs).
    
    Args:
        img (torch.Tensor): Image tensor of shape (B, CH, H, W).
        laf (torch.Tensor): LAFs of shape (B, N, 2, 3).
        PS (int): Patch size. Default is 32.
        normalize_lafs (bool): Whether to normalize the LAFs before extraction. Default is False.
    
    Returns:
        torch.Tensor: Extracted patches of shape (B, N, CH, PS, PS).
    """
    B, CH, H, W = img.shape
    B, N, _, _ = laf.shape
    
    # Normalize LAFs if required
    if normalize_lafs:
        laf[:, :, 0, 0] /= H
        laf[:, :, 0, 1] /= W
        laf[:, :, 1, 0] /= H
        laf[:, :, 1, 1] /= W
        laf[:, :, 0, 2] /= H
        laf[:, :, 1, 2] /= W
    
    # Create image pyramid
    pyramid = [img]
    while pyramid[-1].shape[2] > PS and pyramid[-1].shape[3] > PS:
        pyramid.append(F.interpolate(pyramid[-1], scale_factor=0.5, mode='bilinear', align_corners=False))
    
    # Determine the appropriate pyramid level for each LAF
    scales = torch.sqrt(laf[:, :, 0, 0]**2 + laf[:, :, 0, 1]**2)
    pyramid_levels = torch.clamp(torch.log2(scales * 2), min=0, max=len(pyramid) - 1).long()
    
    # Extract patches
    patches = torch.zeros(B, N, CH, PS, PS, device=img.device)
    for b in range(B):
        for n in range(N):
            level = pyramid_levels[b, n].item()
            current_img = pyramid[level]
            current_H, current_W = current_img.shape[2], current_img.shape[3]
            
            # Transform LAF to the current pyramid level
            scale_factor = 2 ** level
            laf_scaled = laf[b, n].clone()
            laf_scaled[:, 2] *= scale_factor
            
            # Create affine grid
            grid = F.affine_grid(laf_scaled.unsqueeze(0), torch.Size((1, CH, PS, PS)), align_corners=False)
            
            # Sample patch
            patch = F.grid_sample(current_img[b:b+1], grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            patches[b, n] = patch.squeeze(0)
    
    return patches

