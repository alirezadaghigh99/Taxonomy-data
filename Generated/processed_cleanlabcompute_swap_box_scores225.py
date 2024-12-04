import numpy as np

def compute_swap_box_scores(labels, predictions, alpha=0.5, high_probability_threshold=0.9, overlapping_label_check=True, auxiliary_inputs=None):
    def iou(box1, box2):
        """Compute Intersection over Union (IoU) between two bounding boxes."""
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2

        xi1 = max(x1, x1_p)
        yi1 = max(y1, y1_p)
        xi2 = min(x2, x2_p)
        yi2 = min(y2, y2_p)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area != 0 else 0

    def compute_score(label, prediction, alpha, high_probability_threshold):
        """Compute the swap score for a single bounding box."""
        label_box = label['bbox']
        label_class = label['class']
        pred_box = prediction['bbox']
        pred_class = prediction['class']
        pred_prob = prediction['probability']

        iou_score = iou(label_box, pred_box)
        class_match = 1 if label_class == pred_class else 0
        high_confidence = 1 if pred_prob >= high_probability_threshold else 0

        score = alpha * iou_score + (1 - alpha) * (class_match * high_confidence)
        return score

    swap_scores = []

    for i, (label_dict, prediction_array) in enumerate(zip(labels, predictions)):
        image_scores = []
        for label in label_dict:
            max_score = 0
            for prediction in prediction_array:
                score = compute_score(label, prediction, alpha, high_probability_threshold)
                if overlapping_label_check:
                    for other_label in label_dict:
                        if other_label != label and iou(label['bbox'], other_label['bbox']) > 0:
                            score *= 0.5  # Penalize overlapping labels
                max_score = max(max_score, score)
            image_scores.append(max_score)
        swap_scores.append(np.array(image_scores))

    return swap_scores

