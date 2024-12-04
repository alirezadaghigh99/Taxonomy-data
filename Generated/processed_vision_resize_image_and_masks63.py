import torch
import torch.nn.functional as F

def _resize_image_and_masks(image, self_min_size, self_max_size, target=None, fixed_size=None):
    """
    Resizes an image tensor and its corresponding masks, if provided.

    Parameters:
    image (Tensor): A tensor representing the image to be resized, with shape (C, H, W).
    self_min_size (int): The minimum size for the image's smaller dimension when resizing.
    self_max_size (int): The maximum size for the image's larger dimension when resizing.
    target (dict, optional): A dictionary containing additional data, such as masks, that should be resized alongside the image.
                             The dictionary may contain a key "masks" with a tensor of shape (N, H, W).
    fixed_size (tuple, optional): A tuple (height, width) specifying the fixed dimensions to which the image should be resized.

    Returns:
    tuple: A tuple containing:
           - The resized image tensor.
           - The resized target dictionary if it was provided, with resized masks if present.
    """
    def get_size_with_aspect_ratio(image_size, min_size, max_size):
        h, w = image_size
        min_original_size = float(min((h, w)))
        max_original_size = float(max((h, w)))
        
        if max_original_size / min_original_size * min_size > max_size:
            min_size = int(round(max_size * min_original_size / max_original_size))
        
        if (h <= w and h == min_size) or (w <= h and w == min_size):
            return (h, w)
        
        if h < w:
            ow = min_size
            oh = int(min_size * h / w)
        else:
            oh = min_size
            ow = int(min_size * w / h)
        
        return (oh, ow)

    def resize_tensor(tensor, size):
        return F.interpolate(tensor.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)

    if fixed_size is not None:
        size = fixed_size
    else:
        size = get_size_with_aspect_ratio(image.shape[-2:], self_min_size, self_max_size)

    resized_image = resize_tensor(image, size)

    if target is None:
        return resized_image, target

    if "masks" in target:
        masks = target["masks"]
        resized_masks = F.interpolate(masks.unsqueeze(1).float(), size=size, mode='nearest').squeeze(1)
        target["masks"] = resized_masks

    return resized_image, target