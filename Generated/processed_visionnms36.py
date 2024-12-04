import torch
import torchvision.ops as ops
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def nms(boxes, scores, iou_threshold):
    """
    Perform non-maximum suppression (NMS) on a set of bounding boxes.

    Args:
        boxes (Tensor): A tensor of shape [N, 4] representing bounding box coordinates in (x1, y1, x2, y2) format.
        scores (Tensor): A tensor of shape [N] containing scores for each box.
        iou_threshold (float): A float value specifying the IoU threshold for suppression.

    Returns:
        Tensor: An int64 tensor containing the indices of the boxes that have been kept after NMS, sorted in decreasing order of scores.
    """
    # Log API usage
    logger.info("Performing non-maximum suppression (NMS)")

    # Ensure necessary operations are available
    if not hasattr(ops, 'nms'):
        raise RuntimeError("torchvision.ops.nms is not available. Please ensure you have torchvision installed.")

    # Perform NMS
    keep_indices = ops.nms(boxes, scores, iou_threshold)

    # Sort the kept indices by scores in decreasing order
    sorted_indices = keep_indices[torch.argsort(scores[keep_indices], descending=True)]

    return sorted_indices

