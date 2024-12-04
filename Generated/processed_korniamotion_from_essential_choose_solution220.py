import numpy as np
import cv2

def motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2, mask=None):
    """
    Recover the relative camera rotation and translation from an estimated essential matrix.
    
    Parameters:
    - E_mat: Essential matrix (3x3 or batch of 3x3 matrices)
    - K1: Camera matrix for the first camera (3x3)
    - K2: Camera matrix for the second camera (3x3)
    - x1: Points in the first image (Nx2 or batch of Nx2)
    - x2: Corresponding points in the second image (Nx2 or batch of Nx2)
    - mask: Optional mask to exclude points (N or batch of N)
    
    Returns:
    - R: Rotation matrix (3x3)
    - t: Translation vector (3,)
    - points_3D: Triangulated 3D points (Nx3)
    """
    def check_input_shapes():
        if E_mat.shape[-2:] != (3, 3):
            raise ValueError("E_mat must be of shape (3, 3) or (batch_size, 3, 3)")
        if K1.shape != (3, 3) or K2.shape != (3, 3):
            raise ValueError("K1 and K2 must be of shape (3, 3)")
        if x1.shape[-1] != 2 or x2.shape[-1] != 2:
            raise ValueError("x1 and x2 must be of shape (N, 2) or (batch_size, N, 2)")
        if mask is not None and mask.shape[-1] != x1.shape[-2]:
            raise ValueError("mask must be of shape (N) or (batch_size, N)")

    check_input_shapes()
    
    def decompose_essential_matrix(E):
        U, _, Vt = np.linalg.svd(E)
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt
        t = U[:, 2]
        if np.linalg.det(R1) < 0:
            R1 = -R1
        if np.linalg.det(R2) < 0:
            R2 = -R2
        return [(R1, t), (R1, -t), (R2, t), (R2, -t)]
    
    def triangulate_points(P1, P2, x1, x2):
        points_4D = cv2.triangulatePoints(P1, P2, x1.T, x2.T)
        points_3D = points_4D[:3] / points_4D[3]
        return points_3D.T
    
    def count_points_in_front_of_cameras(R, t, x1, x2):
        P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K2 @ np.hstack((R, t.reshape(-1, 1)))
        points_3D = triangulate_points(P1, P2, x1, x2)
        points_3D_hom = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))
        in_front_of_cam1 = points_3D[:, 2] > 0
        in_front_of_cam2 = (R @ points_3D.T + t.reshape(-1, 1))[2, :] > 0
        return np.sum(in_front_of_cam1 & in_front_of_cam2), points_3D
    
    best_count = -1
    best_solution = None
    best_points_3D = None
    
    solutions = decompose_essential_matrix(E_mat)
    for R, t in solutions:
        count, points_3D = count_points_in_front_of_cameras(R, t, x1, x2)
        if count > best_count:
            best_count = count
            best_solution = (R, t)
            best_points_3D = points_3D
    
    if best_solution is None:
        raise ValueError("No valid solution found")
    
    return best_solution[0], best_solution[1], best_points_3D

