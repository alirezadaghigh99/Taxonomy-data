from typing import Union
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms.v2 as tv_tensors

def to_image(input: Union[torch.Tensor, Image.Image, np.ndarray]) -> tv_tensors.Image:
    if isinstance(input, np.ndarray):
        # Convert numpy array to torch tensor
        tensor = torch.from_numpy(input)
        # Ensure the tensor has at least 3 dimensions
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
        else:
            raise ValueError("Unsupported numpy array shape for image conversion.")
    elif isinstance(input, Image.Image):
        # Convert PIL image to torch tensor
        tensor = pil_to_tensor(input)
    elif isinstance(input, torch.Tensor):
        # Input is already a torch tensor
        tensor = input
    else:
        raise TypeError("Input must be a torch.Tensor, PIL.Image.Image, or np.ndarray.")
    
    # Convert the tensor to a tv_tensors.Image object
    return tv_tensors.Image(tensor)

