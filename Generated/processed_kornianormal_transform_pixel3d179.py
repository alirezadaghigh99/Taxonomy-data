import torch

def normal_transform_pixel3d(depth, height, width, eps=1e-6, device='cpu', dtype=torch.float32):
    # Create a 4x4 identity matrix
    transform_matrix = torch.eye(4, device=device, dtype=dtype)
    
    # Adjust the values based on the image dimensions to prevent divide-by-zero errors
    transform_matrix[0, 0] = 2.0 / (width - 1 + eps)
    transform_matrix[1, 1] = 2.0 / (height - 1 + eps)
    transform_matrix[2, 2] = 2.0 / (depth - 1 + eps)
    
    # Adjust the translation part to shift the range from [0, width-1] to [-1, 1]
    transform_matrix[0, 3] = -1.0
    transform_matrix[1, 3] = -1.0
    transform_matrix[2, 3] = -1.0
    
    # Reshape the matrix to (1, 4, 4)
    transform_matrix = transform_matrix.unsqueeze(0)
    
    return transform_matrix

