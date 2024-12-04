import torch

def _torch_svd_cast(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a tensor.")
    
    original_dtype = tensor.dtype
    if original_dtype not in [torch.float32, torch.float64]:
        tensor = tensor.to(torch.float32)
    
    U, S, V = torch.svd(tensor)
    
    return U, S, V

