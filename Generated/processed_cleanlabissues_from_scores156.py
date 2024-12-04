import numpy as np
from typing import Optional, Union

def issues_from_scores(image_scores: np.ndarray, 
                       pixel_scores: Optional[np.ndarray] = None, 
                       threshold: float = None) -> Union[np.ndarray, np.ndarray]:
    # Validate inputs
    if image_scores is None:
        raise ValueError("image_scores cannot be None.")
    if threshold is None or not (0 <= threshold <= 1):
        raise ValueError("threshold must be a float between 0 and 1.")
    
    # Check if pixel_scores is provided
    if pixel_scores is not None:
        if pixel_scores.shape[0] != image_scores.shape[0]:
            raise ValueError("The first dimension of pixel_scores must match the length of image_scores.")
        
        # Create a boolean mask for pixel-level issues
        pixel_issues_mask = pixel_scores < threshold
        return pixel_issues_mask
    else:
        # Identify image indices with scores below the threshold
        image_issues_indices = np.where(image_scores < threshold)[0]
        return image_issues_indices

