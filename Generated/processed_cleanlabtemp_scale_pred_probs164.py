import numpy as np

def temp_scale_pred_probs(pred_probs, temp):
    # Step 1: Clip the probabilities to avoid log(0)
    clipped_probs = np.clip(pred_probs, 1e-15, 1 - 1e-15)
    
    # Step 2: Normalize the probabilities
    clipped_probs /= clipped_probs.sum(axis=1, keepdims=True)
    
    # Step 3: Apply temperature scaling
    log_probs = np.log(clipped_probs)
    scaled_log_probs = log_probs / temp
    
    # Step 4: Re-normalize the probabilities using softmax
    exp_scaled_log_probs = np.exp(scaled_log_probs)
    scaled_probs = exp_scaled_log_probs / exp_scaled_log_probs.sum(axis=1, keepdims=True)
    
    return scaled_probs

