import numpy as np

def _get_label_quality_per_image(pixel_scores, method="softmin", temperature=0.1):
    if pixel_scores is None or len(pixel_scores) == 0:
        raise Exception("pixel_scores cannot be None or empty.")
    
    if temperature is None or temperature == 0:
        raise Exception("temperature must be a non-zero value.")
    
    if method != "softmin":
        raise Exception("Unsupported method. Currently, only the 'softmin' method is supported.")
    
    # Compute the softmin
    exp_scores = np.exp(-pixel_scores / temperature)
    softmin_scores = exp_scores / np.sum(exp_scores)
    
    # The label quality score is the weighted sum of the pixel scores
    label_quality_score = np.sum(softmin_scores * pixel_scores)
    
    return label_quality_score

