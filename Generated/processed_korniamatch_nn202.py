import torch

def _no_match(desc1, desc2):
    B1 = desc1.shape[0]
    return torch.full((B1, 1), float('inf')), torch.full((B1, 2), -1, dtype=torch.long)

def match_nn(desc1, desc2, dm=None):
    # Check if the input descriptors have the correct shapes
    if desc1.ndimension() != 2 or desc2.ndimension() != 2:
        raise ValueError("Input descriptors must be 2D tensors")
    
    B1, D1 = desc1.shape
    B2, D2 = desc2.shape
    
    if D1 != D2:
        raise ValueError("Descriptor dimensions must match")
    
    # If either desc1 or desc2 is empty, return placeholder output
    if B1 == 0 or B2 == 0:
        return _no_match(desc1, desc2)
    
    # Calculate the distance matrix if not provided
    if dm is None:
        dm = torch.cdist(desc1, desc2)
    
    # Find the minimum distances and their corresponding indices in desc2 for each vector in desc1
    min_distances, min_indices = torch.min(dm, dim=1)
    
    # Construct a tensor containing the indices of matching descriptors in desc1 and desc2
    match_indices = torch.stack((torch.arange(B1, dtype=torch.long), min_indices), dim=1)
    
    return min_distances.unsqueeze(1), match_indices

