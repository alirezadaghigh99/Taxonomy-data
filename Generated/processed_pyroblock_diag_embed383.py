import torch

def block_diag_embed(mat):
    """
    Takes a tensor of shape (..., B, M, N) and returns a block diagonal tensor
    of shape (..., B x M, B x N).

    :param torch.Tensor mat: an input tensor with 3 or more dimensions
    :returns torch.Tensor: a block diagonal tensor with dimension `m.dim() - 1`
    """
    # Get the shape of the input tensor
    *batch_dims, B, M, N = mat.shape
    
    # Create an output tensor of zeros with the desired shape
    result_shape = (*batch_dims, B * M, B * N)
    result = torch.zeros(result_shape, dtype=mat.dtype, device=mat.device)
    
    # Fill the block diagonal elements
    for i in range(B):
        result[..., i * M:(i + 1) * M, i * N:(i + 1) * N] = mat[..., i, :, :]
    
    return result

