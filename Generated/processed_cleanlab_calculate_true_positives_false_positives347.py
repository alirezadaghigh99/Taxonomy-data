import numpy as np

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    box1, box2: numpy arrays of shape (4,) representing [x1, y1, x2, y2]
    
    Returns:
    iou: float
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area != 0 else 0
    
    return iou

def _calculate_true_positives_false_positives(pred_bboxes, lab_bboxes, iou_threshold=0.5, return_false_negative=False):
    """
    Calculate true positives (TP) and false positives (FP) for object detection tasks.
    
    Parameters:
    pred_bboxes: numpy array of shape (N, 4) representing predicted bounding boxes
    lab_bboxes: numpy array of shape (M, 4) representing ground truth bounding boxes
    iou_threshold: float, optional, default=0.5
    return_false_negative: bool, optional, default=False
    
    Returns:
    tp: numpy array of shape (N,) representing true positives
    fp: numpy array of shape (N,) representing false positives
    fn: numpy array of shape (M,) representing false negatives (if return_false_negative is True)
    """
    N = pred_bboxes.shape[0]
    M = lab_bboxes.shape[0]
    
    tp = np.zeros(N, dtype=int)
    fp = np.zeros(N, dtype=int)
    fn = np.zeros(M, dtype=int) if return_false_negative else None
    
    if M == 0:
        # If there are no ground truth boxes, all predictions are false positives
        fp[:] = 1
        if return_false_negative:
            return tp, fp, fn
        return tp, fp
    
    iou_matrix = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            iou_matrix[i, j] = calculate_iou(pred_bboxes[i], lab_bboxes[j])
    
    for i in range(N):
        max_iou_idx = np.argmax(iou_matrix[i])
        max_iou = iou_matrix[i, max_iou_idx]
        
        if max_iou >= iou_threshold:
            if fn is not None:
                fn[max_iou_idx] = 0
            tp[i] = 1
        else:
            fp[i] = 1
    
    if return_false_negative:
        for j in range(M):
            if np.max(iou_matrix[:, j]) < iou_threshold:
                fn[j] = 1
        return tp, fp, fn
    
    return tp, fp

