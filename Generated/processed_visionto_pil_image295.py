import numpy as np
from PIL import Image
import torch

def to_pil_image(pic, mode=None):
    if not isinstance(pic, (np.ndarray, torch.Tensor)):
        raise TypeError("Input pic must be a numpy array or a torch tensor.")
    
    if isinstance(pic, torch.Tensor):
        pic = pic.detach().cpu().numpy()
    
    if pic.ndim not in {2, 3}:
        raise ValueError("Input pic must be 2D or 3D.")
    
    if pic.ndim == 2:
        if mode is None:
            mode = "L"
        if mode not in {"L", "I", "I;16", "F"}:
            raise ValueError(f"Unsupported mode {mode} for 1-channel image.")
        return Image.fromarray(pic, mode)
    
    if pic.ndim == 3:
        if pic.shape[2] not in {1, 2, 3, 4}:
            raise ValueError("Input pic must have 1, 2, 3, or 4 channels.")
        
        if pic.shape[2] == 1:
            if mode is None:
                mode = "L"
            if mode not in {"L", "I", "I;16", "F"}:
                raise ValueError(f"Unsupported mode {mode} for 1-channel image.")
            return Image.fromarray(pic[:, :, 0], mode)
        
        if pic.shape[2] == 2:
            if mode is None:
                mode = "LA"
            if mode != "LA":
                raise ValueError(f"Unsupported mode {mode} for 2-channel image.")
            return Image.fromarray(pic, mode)
        
        if pic.shape[2] == 3:
            if mode is None:
                mode = "RGB"
            if mode not in {"RGB", "YCbCr", "HSV"}:
                raise ValueError(f"Unsupported mode {mode} for 3-channel image.")
            return Image.fromarray(pic, mode)
        
        if pic.shape[2] == 4:
            if mode is None:
                mode = "RGBA"
            if mode not in {"RGBA", "CMYK", "RGBX"}:
                raise ValueError(f"Unsupported mode {mode} for 4-channel image.")
            return Image.fromarray(pic, mode)
    
    raise ValueError("Unsupported image format.")

