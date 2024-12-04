from PIL import Image, ImageOps
import numpy as np
import math

def affine(img, angle, translate, scale, shear, interpolation=Image.BILINEAR, fill=0, center=None):
    """
    Apply an affine transformation to an image while keeping the image center invariant.

    Parameters:
    - img: PIL Image or Tensor
    - angle: Rotation angle in degrees
    - translate: Tuple of (horizontal translation, vertical translation)
    - scale: Overall scale factor
    - shear: Tuple of (shear angle x, shear angle y)
    - interpolation: Interpolation mode (default is Image.BILINEAR)
    - fill: Pixel fill value for areas outside the transformed image (default is 0)
    - center: Optional center of rotation (default is the center of the image)

    Returns:
    - Transformed image as a PIL Image or Tensor
    """
    if not isinstance(img, Image.Image):
        raise TypeError("img should be a PIL Image")

    # Get image size
    width, height = img.size

    # Default center is the center of the image
    if center is None:
        center = (width / 2, height / 2)

    # Convert angles from degrees to radians
    angle_rad = math.radians(angle)
    shear_x_rad = math.radians(shear[0])
    shear_y_rad = math.radians(shear[1])

    # Calculate the affine transformation matrix
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    tan_shear_x = math.tan(shear_x_rad)
    tan_shear_y = math.tan(shear_y_rad)

    # Affine transformation matrix components
    a = scale * cos_a
    b = scale * sin_a
    c = -scale * sin_a
    d = scale * cos_a

    # Shear transformation
    a += tan_shear_y * b
    c += tan_shear_y * d
    b += tan_shear_x * a
    d += tan_shear_x * c

    # Translation
    tx = translate[0]
    ty = translate[1]

    # Centering transformation
    cx, cy = center
    tx += cx - a * cx - b * cy
    ty += cy - c * cx - d * cy

    # Affine transformation matrix
    matrix = (a, b, tx, c, d, ty)

    # Apply the affine transformation
    transformed_img = img.transform((width, height), Image.AFFINE, matrix, resample=interpolation, fillcolor=fill)

    return transformed_img

