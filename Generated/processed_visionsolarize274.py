import torch

def _assert_image_tensor(img):
    if not isinstance(img, torch.Tensor):
        raise TypeError("Input image must be a tensor.")
    if img.dtype not in [torch.uint8, torch.float32, torch.float64]:
        raise TypeError("Image tensor must have dtype uint8, float32, or float64.")

def _assert_channels(img):
    if img.shape[-3] not in [1, 3]:
        raise TypeError("Image tensor must have 1 or 3 channels.")

def invert(img):
    if img.dtype == torch.uint8:
        return 255 - img
    else:
        return 1.0 - img

def solarize(img, threshold):
    # Validate the input image tensor
    _assert_image_tensor(img)
    
    # Ensure the image tensor has at least 3 dimensions
    if img.ndim < 3:
        raise TypeError("Image tensor must have at least 3 dimensions.")
    
    # Validate the number of channels
    _assert_channels(img)
    
    # Check if the threshold is valid
    max_value = 255 if img.dtype == torch.uint8 else 1.0
    if threshold > max_value:
        raise TypeError("Threshold value exceeds the maximum value of the image tensor's data type.")
    
    # Invert the image tensor
    inverted_img = invert(img)
    
    # Apply the solarize effect
    solarized_img = torch.where(img >= threshold, inverted_img, img)
    
    return solarized_img

