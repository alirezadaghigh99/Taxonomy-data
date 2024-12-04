from PIL import Image
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import warnings

def resize(img, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True):
    # Check if interpolation is valid
    if not isinstance(interpolation, (InterpolationMode, int)):
        raise TypeError("interpolation must be of type InterpolationMode or an integer corresponding to a Pillow constant.")
    
    # Check if size is valid
    if not (isinstance(size, int) or (isinstance(size, (list, tuple)) and len(size) in [1, 2])):
        raise ValueError("size must be an integer or a list/tuple of length 1 or 2.")
    
    # Check if max_size is provided when size is not a single integer
    if max_size is not None and not isinstance(size, int):
        raise ValueError("max_size can only be used if size is a single integer.")
    
    # Handle PIL Image
    if isinstance(img, Image.Image):
        if not antialias:
            warnings.warn("Antialias is always applied for PIL Images.")
        
        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                if max_size is not None and oh > max_size:
                    oh = max_size
                    ow = int(max_size * w / h)
            else:
                oh = size
                ow = int(size * w / h)
                if max_size is not None and ow > max_size:
                    ow = max_size
                    oh = int(max_size * h / w)
            size = (ow, oh)
        else:
            size = tuple(size)
        
        return img.resize(size, interpolation)
    
    # Handle Tensor
    elif isinstance(img, torch.Tensor):
        if isinstance(size, int):
            h, w = img.shape[-2:]
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                if max_size is not None and oh > max_size:
                    oh = max_size
                    ow = int(max_size * w / h)
            else:
                oh = size
                ow = int(size * w / h)
                if max_size is not None and ow > max_size:
                    ow = max_size
                    oh = int(max_size * h / w)
            size = (oh, ow)
        else:
            size = tuple(size)
        
        return F.resize(img, size, interpolation=interpolation, antialias=antialias)
    
    else:
        raise TypeError("img should be either a PIL Image or a Tensor.")

