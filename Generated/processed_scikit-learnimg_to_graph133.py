import numpy as np
from sklearn.feature_extraction.image import img_to_graph as sklearn_img_to_graph
from scipy.sparse import csr_matrix

def img_to_graph(img, mask=None, return_as=csr_matrix, dtype=np.float64):
    """
    Generates a graph of pixel-to-pixel gradient connections from a 2D or 3D image.

    Parameters:
    - img: array-like of shape (height, width) or (height, width, channel)
        The input image.
    - mask: array-like of shape (height, width), optional
        An optional mask. If provided, only the pixels selected by the mask are connected.
    - return_as: class, optional
        The class to build the adjacency matrix. Default is scipy.sparse.csr_matrix.
    - dtype: data type, optional
        The data type of the returned sparse matrix. Default is np.float64.

    Returns:
    - adjacency_matrix: ndarray or sparse matrix
        The computed adjacency matrix.
    """
    # Ensure the image is a numpy array
    img = np.asarray(img)
    
    # If the image is 3D, we need to reshape it to 2D
    if img.ndim == 3:
        height, width, channels = img.shape
        img = img.reshape((height * width, channels))
    elif img.ndim == 2:
        height, width = img.shape
        img = img.reshape((height * width, 1))
    else:
        raise ValueError("img must be a 2D or 3D array")

    # Generate the adjacency matrix using sklearn's img_to_graph
    adjacency_matrix = sklearn_img_to_graph(img, mask=mask, dtype=dtype)

    # Convert the adjacency matrix to the desired format
    if return_as == np.ndarray:
        adjacency_matrix = adjacency_matrix.toarray()
    else:
        adjacency_matrix = return_as(adjacency_matrix)

    return adjacency_matrix

