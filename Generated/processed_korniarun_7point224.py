import numpy as np

def normalize_points(points):
    """ Normalize points to have zero mean and unit average distance. """
    mean = np.mean(points, axis=1, keepdims=True)
    std = np.std(points, axis=1, keepdims=True)
    normalized_points = (points - mean) / std
    T = np.array([[1/std[0,0], 0, -mean[0,0]/std[0,0]],
                  [0, 1/std[0,1], -mean[0,1]/std[0,1]],
                  [0, 0, 1]])
    return normalized_points, T

def construct_matrix(points1, points2):
    """ Construct the matrix A used in the 7-point algorithm. """
    B, N, _ = points1.shape
    A = np.zeros((B, N, 9))
    for i in range(N):
        x1, y1 = points1[:, i, 0], points1[:, i, 1]
        x2, y2 = points2[:, i, 0], points2[:, i, 1]
        A[:, i, :] = np.stack([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, np.ones(B)], axis=-1)
    return A

def solve_svd(A):
    """ Solve the linear system using SVD. """
    U, S, Vt = np.linalg.svd(A)
    F1 = Vt[:, -1].reshape(-1, 3, 3)
    F2 = Vt[:, -2].reshape(-1, 3, 3)
    return F1, F2

def form_cubic_polynomial(F1, F2):
    """ Form the cubic polynomial det(alpha*F1 + (1-alpha)*F2) = 0. """
    det_poly = np.zeros((F1.shape[0], 4))
    for i in range(F1.shape[0]):
        a0 = np.linalg.det(F2[i])
        a1 = np.linalg.det(F1[i] + F2[i])
        a2 = np.linalg.det(F1[i] - F2[i])
        a3 = np.linalg.det(F1[i])
        det_poly[i] = [a3, a2, a1, a0]
    return det_poly

def solve_cubic(det_poly):
    """ Solve the cubic polynomial for its roots. """
    roots = np.zeros((det_poly.shape[0], 3))
    for i in range(det_poly.shape[0]):
        roots[i] = np.roots(det_poly[i])
    return roots

def compute_fundamental_matrices(F1, F2, roots):
    """ Compute the fundamental matrices from the roots. """
    B = F1.shape[0]
    F_matrices = []
    for i in range(B):
        for root in roots[i]:
            if np.isreal(root):
                alpha = np.real(root)
                F = alpha * F1[i] + (1 - alpha) * F2[i]
                F_matrices.append(F / np.linalg.norm(F))
    return np.array(F_matrices).reshape(B, -1, 3)

def run_7point(points1, points2):
    assert points1.shape[1] == 7 and points2.shape[1] == 7, "Each set of points must contain exactly 7 points."
    assert points1.shape == points2.shape, "Input point sets must have the same shape."
    
    B, N, _ = points1.shape
    
    # Normalize points
    points1, T1 = normalize_points(points1)
    points2, T2 = normalize_points(points2)
    
    # Construct matrix A
    A = construct_matrix(points1, points2)
    
    # Solve using SVD
    F1, F2 = solve_svd(A)
    
    # Form cubic polynomial
    det_poly = form_cubic_polynomial(F1, F2)
    
    # Solve cubic polynomial
    roots = solve_cubic(det_poly)
    
    # Compute fundamental matrices
    F_matrices = compute_fundamental_matrices(F1, F2, roots)
    
    # Denormalize fundamental matrices
    for i in range(F_matrices.shape[0]):
        F_matrices[i] = T2.T @ F_matrices[i] @ T1
    
    return F_matrices

