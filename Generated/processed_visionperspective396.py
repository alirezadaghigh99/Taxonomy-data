from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F

def _get_perspective_coeffs(startpoints, endpoints):
    """Calculate coefficients for perspective transformation."""
    matrix = []
    for p1, p2 in zip(startpoints, endpoints):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(endpoints).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def perspective(image, startpoints, endpoints, interpolation=Image.BILINEAR, fill=None):
    """Perform a perspective transformation on an image."""
    if len(startpoints) != 4 or len(endpoints) != 4:
        raise ValueError("startpoints and endpoints must each contain exactly four points")

    if isinstance(image, Image.Image):
        coeffs = _get_perspective_coeffs(startpoints, endpoints)
        return image.transform(image.size, Image.PERSPECTIVE, coeffs, interpolation, fillcolor=fill)
    
    elif isinstance(image, torch.Tensor):
        if image.ndimension() != 3 and image.ndimension() != 4:
            raise ValueError("Input tensor must be 3D or 4D")
        
        if image.ndimension() == 3:
            image = image.unsqueeze(0)  # Add batch dimension

        # Convert startpoints and endpoints to the format expected by torchvision
        startpoints = torch.tensor(startpoints, dtype=torch.float32).unsqueeze(0)
        endpoints = torch.tensor(endpoints, dtype=torch.float32).unsqueeze(0)

        grid = F.perspective_grid(startpoints, endpoints, image.size()[-2:])
        transformed_image = F.grid_sample(image, grid, mode=interpolation.lower(), padding_mode='zeros' if fill is None else 'border')

        if image.size(0) == 1:
            transformed_image = transformed_image.squeeze(0)  # Remove batch dimension if added

        return transformed_image

    else:
        raise TypeError("Input image must be a PIL Image or a PyTorch Tensor")

