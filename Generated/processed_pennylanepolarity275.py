def polarity(X, Y, kernel, assume_normalized_kernel=False, rescale_class_labels=False, normalize=False):
    """
    Calculate the polarity of a given kernel function.

    Parameters:
    X (list): List of datapoints.
    Y (list): List of class labels of datapoints, assumed to be either -1 or 1.
    kernel (function): Function that maps datapoints to kernel value.
    assume_normalized_kernel (bool, optional): If True, assume the kernel is normalized. Default is False.
    rescale_class_labels (bool, optional): If True, rescale class labels based on the number of datapoints in each class. Default is False.
    normalize (bool, optional): If True, normalize the final polarity value. Default is False.

    Returns:
    float: The kernel polarity.
    """
    n = len(X)
    if n != len(Y):
        raise ValueError("The length of X and Y must be the same.")
    
    # Rescale class labels if the dataset is unbalanced
    if rescale_class_labels:
        pos_count = sum(1 for y in Y if y == 1)
        neg_count = sum(1 for y in Y if y == -1)
        total_count = pos_count + neg_count
        pos_weight = total_count / (2 * pos_count) if pos_count > 0 else 0
        neg_weight = total_count / (2 * neg_count) if neg_count > 0 else 0
        Y = [y * (pos_weight if y == 1 else neg_weight) for y in Y]
    
    # Calculate the polarity
    polarity_value = 0.0
    for i in range(n):
        for j in range(n):
            polarity_value += Y[i] * Y[j] * kernel(X[i], X[j])
    
    # Normalize the polarity value if required
    if normalize:
        if assume_normalized_kernel:
            polarity_value /= n * n
        else:
            norm_factor = sum(kernel(X[i], X[j]) for i in range(n) for j in range(n))
            polarity_value /= norm_factor
    
    return polarity_value

