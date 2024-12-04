import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import normalize

def greedy_coreset(x, indices_unlabeled, indices_labeled, n, distance_metric='euclidean', batch_size=1, normalized=False):
    if normalized:
        x = normalize(x, axis=1)
    
    if distance_metric == 'cosine':
        distance_func = cosine_distances
    elif distance_metric == 'euclidean':
        distance_func = euclidean_distances
    else:
        raise ValueError("Unsupported distance metric. Use 'cosine' or 'euclidean'.")
    
    # Initialize the coreset with the labeled indices
    coreset_indices = list(indices_labeled)
    
    # Compute initial distances from labeled to unlabeled points
    distances = distance_func(x[indices_labeled], x[indices_unlabeled])
    min_distances = np.min(distances, axis=0)
    
    for _ in range(n):
        # Select the batch_size points with the maximum minimum distance to the coreset
        batch_indices = np.argpartition(min_distances, -batch_size)[-batch_size:]
        selected_indices = indices_unlabeled[batch_indices]
        
        # Add selected indices to the coreset
        coreset_indices.extend(selected_indices)
        
        # Remove selected indices from the unlabeled set
        indices_unlabeled = np.delete(indices_unlabeled, batch_indices)
        
        # Update the minimum distances
        new_distances = distance_func(x[selected_indices], x[indices_unlabeled])
        min_distances = np.minimum(min_distances, np.min(new_distances, axis=0))
    
    return np.array(coreset_indices)

