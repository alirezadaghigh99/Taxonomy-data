import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from pathlib import Path

def save_image(tensor, fp, format=None, **kwargs):
    # Check if the input is a list of tensors
    if isinstance(tensor, list):
        # Stack the list of tensors into a single tensor
        tensor = torch.stack(tensor)
    
    # If the tensor is a mini-batch, arrange it into a grid
    if tensor.ndimension() == 4:
        tensor = make_grid(tensor, **kwargs)
    
    # Normalize and clamp the tensor to the [0, 255] range
    tensor = tensor.clone()  # Avoid modifying the original tensor
    tensor = tensor.mul(255).clamp(0, 255).byte()
    
    # Convert the tensor to a NumPy array
    array = tensor.permute(1, 2, 0).cpu().numpy()
    
    # Create a PIL image from the NumPy array
    image = Image.fromarray(array)
    
    # If no format is provided, infer it from the file path
    if format is None:
        if isinstance(fp, (str, Path)):
            format = Path(fp).suffix[1:]  # Get the file extension without the dot
        else:
            raise ValueError("Format must be specified when using a file-like object")
    
    # Save the image
    image.save(fp, format=format)

