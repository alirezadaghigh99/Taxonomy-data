import torch

def _create_identity_grid(size):
    """
    Generates a grid of normalized coordinates for a given image size.
    
    Args:
        size (list of int): The dimensions of the grid (height, width).
        
    Returns:
        torch.Tensor: A tensor containing the grid coordinates.
    """
    height, width = size
    # Create a grid of coordinates
    y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))
    # Stack the coordinates to create a grid
    grid = torch.stack((x, y), 2)
    return grid

