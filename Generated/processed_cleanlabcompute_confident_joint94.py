import numpy as np
from sklearn.calibration import CalibratedClassifierCV

def compute_confident_joint(labels, pred_probs, thresholds=None, calibrate=False, multi_label=False, return_indices_of_off_diagonals=False):
    labels = np.array(labels)
    pred_probs = np.array(pred_probs)
    
    num_classes = pred_probs.shape[1]
    
    if thresholds is None:
        thresholds = np.ones(num_classes) * 0.5
    
    if calibrate:
        # Assuming pred_probs are from a model that can be calibrated
        calibrated_model = CalibratedClassifierCV()
        pred_probs = calibrated_model.fit(pred_probs, labels).predict_proba(pred_probs)
    
    confident_joint = np.zeros((num_classes, num_classes), dtype=int)
    off_diagonal_indices = []

    if multi_label:
        for i in range(len(labels)):
            for j in range(num_classes):
                if pred_probs[i, j] >= thresholds[j]:
                    for k in range(num_classes):
                        if labels[i, k] == 1:
                            confident_joint[k, j] += 1
                            if k != j and return_indices_of_off_diagonals:
                                off_diagonal_indices.append(i)
    else:
        for i in range(len(labels)):
            predicted_label = np.argmax(pred_probs[i])
            if pred_probs[i, predicted_label] >= thresholds[predicted_label]:
                true_label = labels[i]
                confident_joint[true_label, predicted_label] += 1
                if true_label != predicted_label and return_indices_of_off_diagonals:
                    off_diagonal_indices.append(i)
    
    if return_indices_of_off_diagonals:
        return confident_joint, off_diagonal_indices
    else:
        return confident_joint

