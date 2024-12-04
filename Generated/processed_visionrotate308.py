import torch
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

def rotate(img, angle, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0):
    """
    Rotate the image by angle.

    Args:
        img (PIL Image or Tensor): image to be rotated.
        angle (number): rotation angle value in degrees, counter-clockwise.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (sequence, optional): Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.

    Returns:
        PIL Image or Tensor: Rotated image.
    """
    if not isinstance(angle, (int, float)):
        raise TypeError("Argument angle should be int or float")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    if isinstance(interpolation, int):
        interpolation = InterpolationMode(interpolation)
    elif not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant")

    if isinstance(img, Image.Image):
        return img.rotate(angle, resample=interpolation.value, expand=expand, center=center, fillcolor=fill)
    elif isinstance(img, torch.Tensor):
        if interpolation not in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]:
            raise ValueError("Only InterpolationMode.NEAREST and InterpolationMode.BILINEAR are supported for Tensors")
        return F.rotate(img, angle, interpolation=interpolation, expand=expand, center=center, fill=fill)
    else:
        raise TypeError("img should be PIL Image or Tensor")

