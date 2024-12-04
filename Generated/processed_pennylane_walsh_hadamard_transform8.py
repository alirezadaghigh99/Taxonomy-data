import numpy as np
import torch

def _walsh_hadamard_transform(D, n=None):
    """
    Compute the Walsh-Hadamard Transform for a one-dimensional tensor or array D.
    
    Parameters:
    D (torch.Tensor or np.ndarray): Input tensor or array of length 2^n.
    n (int, optional): Number of qubits or size of the transform. If not provided, it is calculated based on the length of D.
    
    Returns:
    torch.Tensor or np.ndarray: Transformed tensor with the same shape as the input.
    """
    if isinstance(D, np.ndarray):
        D = torch.tensor(D, dtype=torch.float32)
    
    length = D.shape[0]
    
    if n is None:
        n = int(np.log2(length))
    
    if length != 2**n:
        raise ValueError("Length of D must be a power of two.")
    
    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.float32) / np.sqrt(2)
    
    def hadamard_transform(tensor, n):
        for i in range(n):
            tensor = tensor.view(-1, 2)
            tensor = torch.matmul(tensor, H)
            tensor = tensor.view(-1)
        return tensor
    
    transformed_D = hadamard_transform(D, n)
    
    if isinstance(D, np.ndarray):
        return transformed_D.numpy()
    return transformed_D

