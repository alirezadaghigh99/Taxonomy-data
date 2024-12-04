import torch

def _box_xywh_to_xyxy(boxes):
    """
    Convert bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2) format.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 4) where N is the number of bounding boxes.
                              Each bounding box is represented as (x, y, w, h).

    Returns:
        torch.Tensor: A tensor of shape (N, 4) where each bounding box is represented as (x1, y1, x2, y2).
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    
    return torch.stack((x1, y1, x2, y2), dim=1)

