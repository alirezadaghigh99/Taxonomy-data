import torch
import cv2
import numpy as np

def undistort_image(image, K, dist):
    """
    Compensates an image for lens distortion.

    Parameters:
    - image (torch.Tensor): Input image with shape (*, C, H, W)
    - K (torch.Tensor): Intrinsic camera matrix with shape (*, 3, 3)
    - dist (torch.Tensor): Distortion coefficients with shape (*, n)

    Returns:
    - torch.Tensor: Undistorted image with the same shape as the input image
    """
    # Ensure the input is a 4D tensor
    assert image.ndim == 4, "Image must be a 4D tensor with shape (*, C, H, W)"
    assert K.ndim == 3 and K.shape[1:] == (3, 3), "K must be a 3D tensor with shape (*, 3, 3)"
    assert dist.ndim == 2, "dist must be a 2D tensor with shape (*, n)"
    
    batch_size, C, H, W = image.shape
    undistorted_images = []

    for i in range(batch_size):
        img_np = image[i].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format and numpy array
        K_np = K[i].cpu().numpy()
        dist_np = dist[i].cpu().numpy()

        # Undistort the image using OpenCV
        undistorted_img_np = cv2.undistort(img_np, K_np, dist_np)

        # Convert back to CHW format and torch.Tensor
        undistorted_img = torch.from_numpy(undistorted_img_np).permute(2, 0, 1)
        undistorted_images.append(undistorted_img)

    # Stack the undistorted images back into a single tensor
    undistorted_images = torch.stack(undistorted_images)

    return undistorted_images

