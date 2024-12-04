import torch
from torch import Tensor
from typing import Tuple, Optional

def match_fginn(
    desc1: Tensor,
    desc2: Tensor,
    lafs1: Tensor,
    lafs2: Tensor,
    th: float = 0.8,
    spatial_th: float = 10.0,
    mutual: bool = False,
    dm: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    # Compute the pairwise distance between descriptors
    if dm is None:
        dm = torch.cdist(desc1, desc2, p=2)
    
    # Find the nearest neighbors in desc2 for each vector in desc1
    min_dist, nn_idx = torch.min(dm, dim=1)
    
    # Apply the descriptor distance threshold
    mask = min_dist <= th
    min_dist = min_dist[mask]
    nn_idx = nn_idx[mask]
    
    # Get the corresponding lafs
    lafs1_matched = lafs1[mask]
    lafs2_matched = lafs2[nn_idx]
    
    # Compute the spatial distance between matched lafs
    spatial_dist = torch.norm(lafs1_matched[:, :2, 2] - lafs2_matched[:, :2, 2], dim=1)
    
    # Apply the spatial distance threshold
    spatial_mask = spatial_dist <= spatial_th
    min_dist = min_dist[spatial_mask]
    nn_idx = nn_idx[spatial_mask]
    idx1 = torch.nonzero(mask)[spatial_mask].squeeze(1)
    
    if mutual:
        # Perform mutual nearest neighbor check
        min_dist2, nn_idx2 = torch.min(dm.t(), dim=1)
        mutual_mask = nn_idx2[nn_idx] == idx1
        min_dist = min_dist[mutual_mask]
        nn_idx = nn_idx[mutual_mask]
        idx1 = idx1[mutual_mask]
    
    return min_dist, idx1, nn_idx

