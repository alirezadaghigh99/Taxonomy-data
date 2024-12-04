import numpy as np
from sklearn.feature_extraction import image

def extract_patches_2d(image_data, patch_size, max_patches=None, random_state=None):
    """
    Extracts patches from a 2D image.

    Parameters:
    - image_data: numpy array of shape (height, width, channels)
    - patch_size: tuple of (patch_height, patch_width)
    - max_patches: int, maximum number of patches to extract (default is None, which extracts all patches)
    - random_state: int or RandomState, optional, random state for sampling

    Returns:
    - patches: numpy array of shape (num_patches, patch_height, patch_width, channels)
    """
    patches = image.extract_patches_2d(image_data, patch_size, max_patches=max_patches, random_state=random_state)
    return patches

