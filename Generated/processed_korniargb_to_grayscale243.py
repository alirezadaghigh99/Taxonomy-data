import torch

def rgb_to_grayscale(image, rgb_weights=None):
    # Validate the input tensor shape
    if image.ndim < 3 or image.shape[-3] != 3:
        raise ValueError("Input image must have shape (*, 3, H, W)")

    # Determine the default weights based on the image data type
    if rgb_weights is None:
        if image.dtype == torch.uint8:
            rgb_weights = torch.tensor([76, 150, 29], dtype=image.dtype, device=image.device) / 255.0
        elif torch.is_floating_point(image):
            rgb_weights = torch.tensor([0.299, 0.587, 0.114], dtype=image.dtype, device=image.device)
        else:
            raise TypeError("Unsupported image data type. Supported types are uint8 and floating-point.")

    # Ensure the weights sum to 1
    if not torch.isclose(rgb_weights.sum(), torch.tensor(1.0, dtype=rgb_weights.dtype, device=rgb_weights.device)):
        raise ValueError("The sum of rgb_weights must be 1")

    # Convert the RGB image to grayscale
    grayscale_image = (image * rgb_weights.view(1, 3, 1, 1)).sum(dim=-3, keepdim=True)

    return grayscale_image

