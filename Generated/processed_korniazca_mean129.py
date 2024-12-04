import torch

def zca_mean(inp, dim, unbiased=True, eps=1e-5, return_inverse=False):
    # Validate input tensor
    if not isinstance(inp, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    if not isinstance(dim, int) or dim < 0 or dim >= inp.ndim:
        raise ValueError("Dimension 'dim' must be a valid dimension of the input tensor")
    if not isinstance(unbiased, bool):
        raise TypeError("Parameter 'unbiased' must be a boolean")
    if not isinstance(eps, (float, int)) or eps <= 0:
        raise ValueError("Parameter 'eps' must be a positive number")
    if not isinstance(return_inverse, bool):
        raise TypeError("Parameter 'return_inverse' must be a boolean")

    # Compute the mean vector along the specified dimension
    mean_vector = inp.mean(dim=dim, keepdim=True)

    # Center the input tensor by subtracting the mean vector
    centered_inp = inp - mean_vector

    # Compute the covariance matrix
    if unbiased:
        cov_matrix = torch.matmul(centered_inp.transpose(dim, -1), centered_inp) / (centered_inp.size(dim) - 1)
    else:
        cov_matrix = torch.matmul(centered_inp.transpose(dim, -1), centered_inp) / centered_inp.size(dim)

    # Perform eigenvalue decomposition on the covariance matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # Compute the ZCA whitening matrix
    D = torch.diag(1.0 / torch.sqrt(eigenvalues + eps))
    zca_matrix = torch.matmul(eigenvectors, torch.matmul(D, eigenvectors.t()))

    # Optionally compute the inverse ZCA matrix
    if return_inverse:
        D_inv = torch.diag(torch.sqrt(eigenvalues + eps))
        zca_matrix_inv = torch.matmul(eigenvectors, torch.matmul(D_inv, eigenvectors.t()))
        return zca_matrix, mean_vector.squeeze(dim), zca_matrix_inv
    else:
        return zca_matrix, mean_vector.squeeze(dim)

