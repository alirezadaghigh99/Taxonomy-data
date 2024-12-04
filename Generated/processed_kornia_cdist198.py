import torch

def _cdist(d1, d2):
    # Check if tensors are compatible with MPS
    def is_mps_compatible(tensor):
        return tensor.device.type == 'mps' and tensor.dtype in [torch.float16, torch.float32]

    if not (is_mps_compatible(d1) and is_mps_compatible(d2)):
        # Use torch.cdist directly if not MPS compatible
        return torch.cdist(d1, d2)

    # Manually compute the Euclidean distances
    # Compute the squared norms of each row in d1 and d2
    d1_squared = torch.sum(d1 ** 2, dim=1, keepdim=True)
    d2_squared = torch.sum(d2 ** 2, dim=1, keepdim=True).t()

    # Compute the pairwise squared distances
    distances_squared = d1_squared + d2_squared - 2 * torch.mm(d1, d2.t())

    # Clamp the distances to a minimum of 0.0 to avoid negative values due to numerical errors
    distances_squared = torch.clamp(distances_squared, min=0.0)

    # Take the square root to get the Euclidean distances
    distances = torch.sqrt(distances_squared)

    return distances

