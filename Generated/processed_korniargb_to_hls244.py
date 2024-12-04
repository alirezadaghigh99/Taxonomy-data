import torch

def rgb_to_hls(image, eps=1e-10):
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a PyTorch tensor.")
    
    if image.ndimension() < 3 or image.size(-3) != 3:
        raise ValueError("Input image must have shape (*, 3, H, W).")
    
    r, g, b = image.unbind(dim=-3)
    
    max_rgb, _ = torch.max(image, dim=-3)
    min_rgb, _ = torch.min(image, dim=-3)
    
    l = (max_rgb + min_rgb) / 2
    
    delta = max_rgb - min_rgb
    s = torch.where(l < 0.5, delta / (max_rgb + min_rgb + eps), delta / (2 - max_rgb - min_rgb + eps))
    
    delta_r = (((max_rgb - r) / 6) + (delta / 2)) / (delta + eps)
    delta_g = (((max_rgb - g) / 6) + (delta / 2)) / (delta + eps)
    delta_b = (((max_rgb - b) / 6) + (delta / 2)) / (delta + eps)
    
    h = torch.zeros_like(max_rgb)
    h[max_rgb == r] = delta_b[max_rgb == r] - delta_g[max_rgb == r]
    h[max_rgb == g] = (1 / 3) + delta_r[max_rgb == g] - delta_b[max_rgb == g]
    h[max_rgb == b] = (2 / 3) + delta_g[max_rgb == b] - delta_r[max_rgb == b]
    
    h = (h + 1) % 1  # Ensure hue is in the range [0, 1]
    
    hls = torch.stack([h, l, s], dim=-3)
    
    return hls

