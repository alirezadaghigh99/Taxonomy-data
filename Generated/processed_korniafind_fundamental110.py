import torch

def find_fundamental(points1, points2, weights, method):
    """
    Compute the fundamental matrix using the specified method.

    Args:
        points1: A set of points in the first image with a tensor shape (B, N, 2), N>=8.
        points2: A set of points in the second image with a tensor shape (B, N, 2), N>=8.
        weights: Tensor containing the weights per point correspondence with a shape of (B, N).
        method: The method to use for computing the fundamental matrix. Supported methods are "7POINT" and "8POINT".

    Returns:
        The computed fundamental matrix with shape (B, 3*m, 3), where `m` is the number of fundamental matrices.

    Raises:
        ValueError: If an invalid method is provided.
    """
    if method not in ["7POINT", "8POINT"]:
        raise ValueError(f"Invalid method '{method}'. Supported methods are '7POINT' and '8POINT'.")

    B, N, _ = points1.shape
    if N < 8:
        raise ValueError("At least 8 points are required to compute the fundamental matrix.")

    # Normalize the points
    def normalize_points(points):
        mean = points.mean(dim=1, keepdim=True)
        std = points.std(dim=1, keepdim=True)
        return (points - mean) / std

    points1_normalized = normalize_points(points1)
    points2_normalized = normalize_points(points2)

    # Construct the matrix A for the linear system
    def construct_A(p1, p2):
        x1, y1 = p1[:, :, 0], p1[:, :, 1]
        x2, y2 = p2[:, :, 0], p2[:, :, 1]
        A = torch.stack([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, torch.ones_like(x1)], dim=-1)
        return A

    A = construct_A(points1_normalized, points2_normalized)

    # Apply weights
    W = weights.unsqueeze(-1)
    A = A * W

    # Solve the linear system using SVD
    U, S, Vt = torch.svd(A)
    F = Vt[:, -1].view(B, 3, 3)

    if method == "7POINT":
        # For 7-point algorithm, we need to enforce the rank-2 constraint
        U, S, Vt = torch.svd(F)
        S[:, -1] = 0
        F = U @ torch.diag_embed(S) @ Vt

    # Denormalize the fundamental matrix
    def denormalize_F(F, p1, p2):
        mean1, std1 = p1.mean(dim=1, keepdim=True), p1.std(dim=1, keepdim=True)
        mean2, std2 = p2.mean(dim=1, keepdim=True), p2.std(dim=1, keepdim=True)
        T1 = torch.diag_embed(torch.tensor([std1[:, 0, 0], std1[:, 0, 1], torch.ones(B)]).to(F.device))
        T1[:, 0, 2] = mean1[:, 0, 0]
        T1[:, 1, 2] = mean1[:, 0, 1]
        T2 = torch.diag_embed(torch.tensor([std2[:, 0, 0], std2[:, 0, 1], torch.ones(B)]).to(F.device))
        T2[:, 0, 2] = mean2[:, 0, 0]
        T2[:, 1, 2] = mean2[:, 0, 1]
        F = T2.transpose(1, 2) @ F @ T1
        return F

    F = denormalize_F(F, points1, points2)

    return F

