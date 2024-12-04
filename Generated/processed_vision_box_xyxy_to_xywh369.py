import torch

def _box_xyxy_to_xywh(boxes):
    """
    Convert bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h) format.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 4) where N is the number of bounding boxes.
                              Each bounding box is represented as (x1, y1, x2, y2).

    Returns:
        torch.Tensor: A tensor of shape (N, 4) where each bounding box is represented as (x, y, w, h).
    """
    # Ensure the input is a tensor
    if not isinstance(boxes, torch.Tensor):
        raise TypeError("Input boxes should be a torch.Tensor")

    # Extract the coordinates
    x1, y1, x2, y2 = boxes.unbind(dim=1)

    # Calculate the width and height
    w = x2 - x1
    h = y2 - y1

    # Stack the new coordinates
    xywh_boxes = torch.stack((x1, y1, w, h), dim=1)

    return xywh_boxes

