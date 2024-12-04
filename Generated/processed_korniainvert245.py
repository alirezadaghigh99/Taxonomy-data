import torch

def invert(img, max_val=None):
    # Check if the input image is a tensor
    assert isinstance(img, torch.Tensor), "Input image must be a tensor"
    
    # If max_val is not provided, use the maximum value in the img tensor
    if max_val is None:
        max_val = img.max()
    else:
        # Check if the max_val is a tensor
        assert isinstance(max_val, torch.Tensor), "Maximum value must be a tensor"
    
    # Invert the image tensor
    inverted_img = max_val - img
    
    return inverted_img

