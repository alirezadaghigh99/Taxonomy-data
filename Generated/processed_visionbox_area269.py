import torch

def box_area(boxes):
    """
    Compute the area of a set of bounding boxes.
    
    Args:
        boxes (torch.Tensor): A tensor of shape (N, 4) containing the coordinates of the boxes
                              in (x1, y1, x2, y2) format.
    
    Returns:
        torch.Tensor: A tensor of shape (N) containing the area for each box.
    """
    # Ensure the input is a tensor
    if not isinstance(boxes, torch.Tensor):
        raise TypeError("Input boxes must be a torch.Tensor")
    
    # Check the shape of the input tensor
    if boxes.ndimension() != 2 or boxes.size(1) != 4:
        raise ValueError("Input boxes must have shape (N, 4)")
    
    # Extract the coordinates
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    # Compute the width and height of each box
    widths = x2 - x1
    heights = y2 - y1
    
    # Compute the area of each box
    areas = widths * heights
    
    return areas

