import torch

def _box_cxcywh_to_xyxy(boxes):
    """
    Convert bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 4) where N is the number of bounding boxes.
                              Each box is represented as (cx, cy, w, h).

    Returns:
        torch.Tensor: A tensor of shape (N, 4) where each box is represented as (x1, y1, x2, y2).
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2), dim=-1)

