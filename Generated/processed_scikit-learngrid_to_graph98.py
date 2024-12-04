import numpy as np
from scipy.sparse import coo_matrix

def pixel_to_pixel_graph(n_x, n_y, n_z=1, mask=None, return_as=coo_matrix, dtype=int):
    # Total number of voxels
    n_voxels = n_x * n_y * n_z
    
    # Create a mask if none is provided
    if mask is None:
        mask = np.ones((n_x, n_y, n_z), dtype=bool)
    
    # Ensure the mask is boolean
    mask = mask.astype(bool)
    
    # Get the indices of the voxels that are within the mask
    indices = np.argwhere(mask)
    
    # Initialize lists to store the row and column indices of the adjacency matrix
    row_indices = []
    col_indices = []
    
    # Define the possible 6-connectivity (for 3D) or 4-connectivity (for 2D) neighbors
    neighbors = [
        (1, 0, 0), (-1, 0, 0),  # x-axis neighbors
        (0, 1, 0), (0, -1, 0),  # y-axis neighbors
        (0, 0, 1), (0, 0, -1)   # z-axis neighbors (only for 3D)
    ]
    
    for idx in indices:
        x, y, z = idx
        current_index = x * n_y * n_z + y * n_z + z
        
        for dx, dy, dz in neighbors:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < n_x and 0 <= ny < n_y and 0 <= nz < n_z and mask[nx, ny, nz]:
                neighbor_index = nx * n_y * n_z + ny * n_z + nz
                row_indices.append(current_index)
                col_indices.append(neighbor_index)
    
    # Create the adjacency matrix
    data = np.ones(len(row_indices), dtype=dtype)
    adjacency_matrix = return_as((data, (row_indices, col_indices)), shape=(n_voxels, n_voxels))
    
    return adjacency_matrix

