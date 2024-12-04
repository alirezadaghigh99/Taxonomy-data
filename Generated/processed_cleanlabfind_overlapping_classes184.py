import numpy as np
import pandas as pd

def find_overlapping_classes(labels=None, pred_probs=None, asymmetric=False, class_names=None, num_examples=None, joint=None, confident_joint=None):
    if joint is None and confident_joint is None:
        if labels is None or pred_probs is None:
            raise ValueError("You must provide either (labels and pred_probs) or (joint and num_examples) or (confident_joint).")
        
        # Compute the confident joint if not provided
        confident_joint = compute_confident_joint(labels, pred_probs)
    
    if joint is None:
        joint = confident_joint / np.sum(confident_joint)
    
    if num_examples is None:
        num_examples = np.sum(confident_joint)
    
    overlapping_classes = []
    K = joint.shape[0]
    
    for i in range(K):
        for j in range(i + 1, K):
            if asymmetric:
                num_overlap_ij = confident_joint[i, j]
                num_overlap_ji = confident_joint[j, i]
                joint_prob_ij = joint[i, j]
                joint_prob_ji = joint[j, i]
                
                if num_overlap_ij > 0:
                    overlapping_classes.append([i, j, num_overlap_ij, joint_prob_ij])
                if num_overlap_ji > 0:
                    overlapping_classes.append([j, i, num_overlap_ji, joint_prob_ji])
            else:
                num_overlap = confident_joint[i, j] + confident_joint[j, i]
                joint_prob = joint[i, j] + joint[j, i]
                
                if num_overlap > 0:
                    overlapping_classes.append([i, j, num_overlap, joint_prob])
    
    df = pd.DataFrame(overlapping_classes, columns=["Class Index A", "Class Index B", "Num Overlapping Examples", "Joint Probability"])
    df = df.sort_values(by="Joint Probability", ascending=False).reset_index(drop=True)
    
    if class_names is not None:
        df["Class Name A"] = df["Class Index A"].apply(lambda x: class_names[x])
        df["Class Name B"] = df["Class Index B"].apply(lambda x: class_names[x])
    
    return df

def compute_confident_joint(labels, pred_probs):
    # Placeholder function to compute the confident joint
    # This should be replaced with the actual implementation
    K = pred_probs.shape[1]
    confident_joint = np.zeros((K, K))
    
    for i in range(len(labels)):
        true_label = labels[i]
        pred_label = np.argmax(pred_probs[i])
        confident_joint[true_label, pred_label] += 1
    
    return confident_joint