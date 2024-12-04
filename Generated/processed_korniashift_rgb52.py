import torch

def shift_rgb(image, r_shift, g_shift, b_shift):
    # Verify that the image is a valid tensor and a color image
    if not isinstance(image, torch.Tensor):
        raise TypeError("Image must be a torch.Tensor")
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Image must have shape (N, 3, H, W)")

    if not isinstance(r_shift, torch.Tensor) or not isinstance(g_shift, torch.Tensor) or not isinstance(b_shift, torch.Tensor):
        raise TypeError("Shift values must be torch.Tensors")
    if r_shift.ndim != 1 or g_shift.ndim != 1 or b_shift.ndim != 1:
        raise ValueError("Shift values must have shape (N)")

    N, C, H, W = image.shape

    if r_shift.shape[0] != N or g_shift.shape[0] != N or b_shift.shape[0] != N:
        raise ValueError("Shift values must have the same batch size as the image")

    # Apply the shifts to each channel
    shifted_image = image.clone()
    for i in range(N):
        shifted_image[i, 0, :, :] += r_shift[i]
        shifted_image[i, 1, :, :] += g_shift[i]
        shifted_image[i, 2, :, :] += b_shift[i]

    # Clamp the output values between 0 and 1
    shifted_image = torch.clamp(shifted_image, 0, 1)

    return shifted_image

