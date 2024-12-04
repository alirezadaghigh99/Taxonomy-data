import logging
from PIL import Image
import torch
import torchvision.transforms.functional as F_t
import torchvision.transforms.functional_pil as F_pil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_dimensions(img):
    """
    Get the dimensions of an image.

    Parameters:
    img (PIL.Image.Image or torch.Tensor): The input image.

    Returns:
    list: A list of integers in the format [channels, height, width].
    """
    # Check if the input is a PIL Image
    if isinstance(img, Image.Image):
        dimensions = F_pil.get_dimensions(img)
    # Check if the input is a Tensor
    elif isinstance(img, torch.Tensor):
        dimensions = F_t.get_dimensions(img)
    else:
        raise TypeError("Input must be a PIL Image or a torch.Tensor")

    # Log API usage if not in scripting or tracing mode
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        logger.info(f"API get_dimensions called with input type: {type(img).__name__}")

    return dimensions

