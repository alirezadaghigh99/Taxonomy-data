import torch

def match_snn(desc1, desc2, th=0.8, dm=None):
    """
    Matches descriptors from desc1 to desc2 using the second nearest neighbor ratio test.

    Args:
        desc1 (torch.Tensor): Tensor of shape (B1, D) representing descriptors from the first set.
        desc2 (torch.Tensor): Tensor of shape (B2, D) representing descriptors from the second set.
        th (float): Threshold for the ratio test. Default is 0.8.
        dm (torch.Tensor, optional): Precomputed distance matrix of shape (B1, B2). If None, it will be computed.

    Returns:
        tuple: (distances, indices) where:
            - distances (torch.Tensor): Tensor of shape (B3, 1) with distances of matching descriptors.
            - indices (torch.Tensor): Long tensor of shape (B3, 2) with indices of matching descriptors in desc1 and desc2.
    """
    if desc2.size(0) < 2:
        return torch.empty((0, 1)), torch.empty((0, 2), dtype=torch.long)

    if dm is None:
        dm = torch.cdist(desc1, desc2, p=2)  # Compute the distance matrix if not provided

    # Get the two smallest distances and their indices
    top2_distances, top2_indices = torch.topk(dm, k=2, largest=False, dim=1)

    # Compute the ratio of the smallest distance to the second smallest distance
    ratio = top2_distances[:, 0] / top2_distances[:, 1]

    # Find the matches that satisfy the ratio test
    valid_matches = ratio <= th

    if not valid_matches.any():
        return torch.empty((0, 1)), torch.empty((0, 2), dtype=torch.long)

    # Get the distances and indices of the valid matches
    distances = top2_distances[valid_matches, 0].unsqueeze(1)
    indices = torch.stack((torch.arange(desc1.size(0))[valid_matches], top2_indices[valid_matches, 0]), dim=1)

    return distances, indices

