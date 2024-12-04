import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def create_knn_graph_and_index(features, n_neighbors=5, metric='euclidean', correct_duplicates=False, **kwargs):
    """
    Create a KNN graph and a KNN search index from input features.

    Parameters:
    - features (array-like): Input feature array of shape (n_samples, n_features).
    - n_neighbors (int): Number of nearest neighbors to use. Default is 5.
    - metric (str): Distance metric to use. Default is 'euclidean'.
    - correct_duplicates (bool): Whether to correct exact duplicates. Default is False.
    - **kwargs: Additional keyword arguments for the NearestNeighbors constructor.

    Returns:
    - tuple: (sparse, weighted adjacency matrix, KNN search object)
    """
    # Initialize the NearestNeighbors object
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, **kwargs)
    
    # Fit the KNN model
    knn.fit(features)
    
    # Find the k-neighbors for each point
    distances, indices = knn.kneighbors(features)
    
    # Correct for exact duplicates if required
    if correct_duplicates:
        for i in range(len(features)):
            unique_indices = np.unique(indices[i])
            if len(unique_indices) < n_neighbors:
                additional_indices = np.setdiff1d(np.arange(len(features)), unique_indices)
                indices[i, len(unique_indices):] = additional_indices[:n_neighbors - len(unique_indices)]
                distances[i, len(unique_indices):] = np.inf  # Assign a large distance to these additional points
    
    # Create the sparse adjacency matrix
    n_samples = features.shape[0]
    row_indices = np.repeat(np.arange(n_samples), n_neighbors)
    col_indices = indices.flatten()
    data = distances.flatten()
    
    adjacency_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(n_samples, n_samples))
    
    return adjacency_matrix, knn

