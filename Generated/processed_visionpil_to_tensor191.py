import torch
import numpy as np
from PIL import Image
import copy

def pil_to_tensor(pic):
    if not isinstance(pic, Image.Image):
        raise TypeError(f"Input should be a PIL Image, but got {type(pic)}")
    
    # Check if the image is an accimage Image
    if hasattr(pic, 'accimage'):
        # Convert accimage to numpy array and then to tensor
        np_array = np.array(pic, copy=True)
        tensor = torch.from_numpy(np_array).type(torch.uint8)
        return tensor
    
    # Convert PIL Image to numpy array
    np_array = np.array(pic, copy=True)
    
    # Convert numpy array to tensor
    tensor = torch.from_numpy(np_array)
    
    # If the image has an alpha channel, it will have 4 channels (RGBA)
    # Otherwise, it will have 3 channels (RGB)
    if len(tensor.shape) == 3 and tensor.shape[2] in [3, 4]:
        # Rearrange dimensions from HWC to CHW
        tensor = tensor.permute(2, 0, 1)
    
    return tensor

