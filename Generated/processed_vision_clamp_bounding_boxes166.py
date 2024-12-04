import torch
from enum import Enum

class BoundingBoxFormat(Enum):
    XYXY = 1
    XYWH = 2
    CXCYWH = 3

def _convert_to_xyxy(bounding_boxes, format):
    if format == BoundingBoxFormat.XYXY:
        return bounding_boxes
    elif format == BoundingBoxFormat.XYWH:
        x, y, w, h = bounding_boxes[:, 0], bounding_boxes[:, 1], bounding_boxes[:, 2], bounding_boxes[:, 3]
        return torch.stack([x, y, x + w, y + h], dim=1)
    elif format == BoundingBoxFormat.CXCYWH:
        cx, cy, w, h = bounding_boxes[:, 0], bounding_boxes[:, 1], bounding_boxes[:, 2], bounding_boxes[:, 3]
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
    else:
        raise ValueError("Unsupported bounding box format")

def _convert_from_xyxy(bounding_boxes, format):
    if format == BoundingBoxFormat.XYXY:
        return bounding_boxes
    elif format == BoundingBoxFormat.XYWH:
        x1, y1, x2, y2 = bounding_boxes[:, 0], bounding_boxes[:, 1], bounding_boxes[:, 2], bounding_boxes[:, 3]
        return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=1)
    elif format == BoundingBoxFormat.CXCYWH:
        x1, y1, x2, y2 = bounding_boxes[:, 0], bounding_boxes[:, 1], bounding_boxes[:, 2], bounding_boxes[:, 3]
        return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=1)
    else:
        raise ValueError("Unsupported bounding box format")

def _clamp_bounding_boxes(bounding_boxes, format, canvas_size):
    # Convert to XYXY format
    bounding_boxes_xyxy = _convert_to_xyxy(bounding_boxes, format)
    
    # Clamp the coordinates
    height, width = canvas_size
    bounding_boxes_xyxy[:, 0] = torch.clamp(bounding_boxes_xyxy[:, 0], 0, width)
    bounding_boxes_xyxy[:, 1] = torch.clamp(bounding_boxes_xyxy[:, 1], 0, height)
    bounding_boxes_xyxy[:, 2] = torch.clamp(bounding_boxes_xyxy[:, 2], 0, width)
    bounding_boxes_xyxy[:, 3] = torch.clamp(bounding_boxes_xyxy[:, 3], 0, height)
    
    # Convert back to the original format
    clamped_bounding_boxes = _convert_from_xyxy(bounding_boxes_xyxy, format)
    
    return clamped_bounding_boxes

