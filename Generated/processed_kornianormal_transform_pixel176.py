import torch

def normal_transform_pixel(height, width, eps=1e-14, device=None, dtype=None):
    # Create the transformation matrix
    transform_matrix = torch.tensor([
        [2.0 / (width - eps), 0.0, -1.0],
        [0.0, 2.0 / (height - eps), -1.0],
        [0.0, 0.0, 1.0]
    ], device=device, dtype=dtype)
    
    # Add an additional dimension of size 1 at the beginning
    transform_matrix = transform_matrix.unsqueeze(0)
    
    return transform_matrix

