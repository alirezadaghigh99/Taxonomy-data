import torch

def get_sobel_kernel2d(device='cpu', dtype=torch.float32):
    # Define the Sobel kernel for the x direction
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=dtype, device=device)
    
    # Define the Sobel kernel for the y direction
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=dtype, device=device)
    
    # Stack the kernels into a single tensor
    sobel_kernel = torch.stack([sobel_x, sobel_y])
    
    return sobel_kernel

