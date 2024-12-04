import numpy as np
import torch

def scale_intrinsics(camera_matrix, scale_factor):
    """
    Scales the focal length and center of projection in the camera matrix by the given scale factor.

    Parameters:
    camera_matrix (torch.Tensor or np.ndarray): The camera matrix with shape (B, 3, 3).
    scale_factor (float or torch.Tensor or np.ndarray): The scale factor to apply.

    Returns:
    torch.Tensor or np.ndarray: The scaled camera matrix with the same shape as the input (B, 3, 3).
    """
    if isinstance(camera_matrix, np.ndarray):
        if isinstance(scale_factor, (float, int)):
            scale_factor = np.array(scale_factor)
        elif isinstance(scale_factor, np.ndarray):
            scale_factor = scale_factor.reshape(-1, 1, 1)
        else:
            raise TypeError("Scale factor must be a float, int, or np.ndarray when camera_matrix is an np.ndarray.")
        
        scaled_matrix = camera_matrix.copy()
        scaled_matrix[:, 0, 0] *= scale_factor
        scaled_matrix[:, 1, 1] *= scale_factor
        scaled_matrix[:, 0, 2] *= scale_factor
        scaled_matrix[:, 1, 2] *= scale_factor

    elif isinstance(camera_matrix, torch.Tensor):
        if isinstance(scale_factor, (float, int)):
            scale_factor = torch.tensor(scale_factor, dtype=camera_matrix.dtype, device=camera_matrix.device)
        elif isinstance(scale_factor, torch.Tensor):
            scale_factor = scale_factor.view(-1, 1, 1)
        else:
            raise TypeError("Scale factor must be a float, int, or torch.Tensor when camera_matrix is a torch.Tensor.")
        
        scaled_matrix = camera_matrix.clone()
        scaled_matrix[:, 0, 0] *= scale_factor
        scaled_matrix[:, 1, 1] *= scale_factor
        scaled_matrix[:, 0, 2] *= scale_factor
        scaled_matrix[:, 1, 2] *= scale_factor

    else:
        raise TypeError("camera_matrix must be either a numpy.ndarray or a torch.Tensor.")

    return scaled_matrix

