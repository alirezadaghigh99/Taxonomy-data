import torch
from torchvision import transforms
from PIL import Image, ExifTags

class ImageReadMode:
    RGB = 'RGB'
    GRAYSCALE = 'L'

def read_image(path, mode=ImageReadMode.RGB, apply_exif_orientation=True):
    """
    Reads a JPEG, PNG, or GIF image from the specified path and converts it into a 3-dimensional
    RGB or grayscale Tensor with values in the range of uint8 [0, 255].

    Parameters:
    - path (str): The file path to the image.
    - mode (ImageReadMode, optional): The mode to read the image in. Can be 'RGB' or 'L' (grayscale).
      Default is 'RGB'.
    - apply_exif_orientation (bool, optional): Whether to apply the EXIF orientation tag to the image.
      Default is True.

    Returns:
    - torch.Tensor: A 3-dimensional Tensor representing the image.
    """
    # Open the image file
    with Image.open(path) as img:
        # Apply EXIF orientation if needed
        if apply_exif_orientation:
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = img._getexif()
                if exif is not None:
                    orientation = exif.get(orientation)
                    if orientation == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation == 8:
                        img = img.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                # If the image does not have EXIF data, continue without applying orientation
                pass

        # Convert image to the specified mode
        img = img.convert(mode)

        # Convert image to tensor
        transform = transforms.ToTensor()
        img_tensor = transform(img)

        # Convert tensor to uint8 range [0, 255]
        img_tensor = (img_tensor * 255).to(torch.uint8)

        return img_tensor

