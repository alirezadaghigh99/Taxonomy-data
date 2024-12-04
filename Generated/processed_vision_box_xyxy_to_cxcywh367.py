import torch

def _box_xyxy_to_cxcywh(boxes):
    """
    Convert bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.

    Args:
        boxes (Tensor): A tensor of shape (N, 4) where N is the number of bounding boxes.
                        Each box is represented as (x1, y1, x2, y2).

    Returns:
        Tensor: A tensor of shape (N, 4) where each box is represented as (cx, cy, w, h).
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, w, h), dim=-1)

