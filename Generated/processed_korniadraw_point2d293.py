import numpy as np

def draw_point2d(image, points, color):
    """
    Draw points on a 2D image tensor.

    Parameters:
    - image: numpy array of shape (H, W) for grayscale or (C, H, W) for multi-channel.
    - points: list of tuples [(x1, y1), (x2, y2), ...] specifying the coordinates to color.
    - color: numpy array of shape (C,) for multi-channel or a scalar for grayscale.

    Returns:
    - Modified image with points colored.
    """
    if len(image.shape) == 2:
        # Grayscale image
        H, W = image.shape
        for (x, y) in points:
            if 0 <= x < W and 0 <= y < H:
                image[y, x] = color
    elif len(image.shape) == 3:
        # Multi-channel image
        C, H, W = image.shape
        for (x, y) in points:
            if 0 <= x < W and 0 <= y < H:
                image[:, y, x] = color
    else:
        raise ValueError("Unsupported image shape: {}".format(image.shape))
    
    return image

