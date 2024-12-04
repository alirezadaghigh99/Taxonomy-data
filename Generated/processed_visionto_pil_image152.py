import numpy as np
from PIL import Image
import torch

def to_pil_image(pic, mode=None):
    """
    Convert a tensor or an ndarray to a PIL Image.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (str, optional): Mode to be used for the PIL Image.

    Returns:
        PIL.Image: Image converted to PIL format.
    """
    if not (isinstance(pic, torch.Tensor) or isinstance(pic, np.ndarray)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    if isinstance(pic, torch.Tensor):
        if pic.ndimension() not in {2, 3}:
            raise ValueError('pic should be 2 or 3 dimensional. Got {} dimensions.'.format(pic.ndimension()))
        pic = pic.cpu().numpy()

    if isinstance(pic, np.ndarray):
        if pic.ndim not in {2, 3}:
            raise ValueError('pic should be 2 or 3 dimensional. Got {} dimensions.'.format(pic.ndim))

    if pic.ndim == 2:
        # Grayscale image
        mode = 'L' if mode is None else mode
    elif pic.ndim == 3:
        if pic.shape[2] == 1:
            # Grayscale image with a single channel
            pic = pic[:, :, 0]
            mode = 'L' if mode is None else mode
        elif pic.shape[2] == 3:
            # RGB image
            mode = 'RGB' if mode is None else mode
        elif pic.shape[2] == 4:
            # RGBA image
            mode = 'RGBA' if mode is None else mode
        else:
            raise ValueError('pic should have 1, 3 or 4 channels. Got {} channels.'.format(pic.shape[2]))

    if mode not in ['L', 'RGB', 'RGBA']:
        raise ValueError('Unsupported mode: {}. Supported modes are L, RGB, RGBA.'.format(mode))

    return Image.fromarray(pic, mode=mode)

