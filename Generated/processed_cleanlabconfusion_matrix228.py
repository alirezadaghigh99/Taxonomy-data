import numpy as np

def confusion_matrix(true, pred):
    """
    Compute the confusion matrix to evaluate the accuracy of a classification.

    Parameters
    ----------
    true : np.ndarray 1d
        Contains true labels.
        Assumes true and pred contains the same set of distinct labels.

    pred : np.ndarray 1d
        A discrete vector of predicted labels, i.e. some labels may be erroneous.
        *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    Returns
    -------
    confusion_matrix : np.ndarray (2D)
        Matrix of confusion counts with true on rows and pred on columns.
    """
    # Ensure inputs are numpy arrays
    true = np.asarray(true)
    pred = np.asarray(pred)
    
    # Check if true and pred have the same length
    if true.shape[0] != pred.shape[0]:
        raise ValueError("true and pred must be the same length")
    
    # Get the number of classes
    num_classes = len(np.unique(true))
    
    # Initialize the confusion matrix with zeros
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # Populate the confusion matrix
    for t, p in zip(true, pred):
        conf_matrix[t, p] += 1
    
    return conf_matrix

