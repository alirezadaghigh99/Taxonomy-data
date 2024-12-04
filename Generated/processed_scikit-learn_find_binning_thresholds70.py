import numpy as np

def _find_binning_thresholds(col_data, max_bins):
    """
    Extracts quantiles from a continuous feature to determine binning thresholds.

    Parameters:
    col_data (array-like): The continuous feature to bin.
    max_bins (int): The maximum number of bins to use for non-missing values.

    Returns:
    ndarray: An array of shape (min(max_bins, n_unique_values) - 1) containing increasing numeric values
             that can be used to separate the bins.
    """
    # Convert col_data to a numpy array and remove missing values
    col_data = np.array(col_data)
    col_data = col_data[~np.isnan(col_data)]
    
    # Sort the data and identify distinct values
    sorted_data = np.sort(col_data)
    unique_values = np.unique(sorted_data)
    
    n_unique_values = len(unique_values)
    
    if n_unique_values <= max_bins:
        # Calculate midpoints between consecutive distinct values
        midpoints = (unique_values[:-1] + unique_values[1:]) / 2
    else:
        # Compute approximate midpoint percentiles
        percentiles = np.linspace(0, 100, num=max_bins + 1)
        bin_edges = np.percentile(sorted_data, percentiles)
        midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Ensure no +inf thresholds
    midpoints = np.clip(midpoints, a_min=None, a_max=np.inf)
    
    return midpoints

