def _find_binning_thresholds(col_data, max_bins):
    """Extract quantiles from a continuous feature.

    Missing values are ignored for finding the thresholds.

    Parameters
    ----------
    col_data : array-like, shape (n_samples,)
        The continuous feature to bin.
    max_bins: int
        The maximum number of bins to use for non-missing values. If for a
        given feature the number of unique values is less than ``max_bins``,
        then those unique values will be used to compute the bin thresholds,
        instead of the quantiles

    Return
    ------
    binning_thresholds : ndarray of shape(min(max_bins, n_unique_values) - 1,)
        The increasing numeric values that can be used to separate the bins.
        A given value x will be mapped into bin value i iff
        bining_thresholds[i - 1] < x <= binning_thresholds[i]
    """
    # ignore missing values when computing bin thresholds
    missing_mask = np.isnan(col_data)
    if missing_mask.any():
        col_data = col_data[~missing_mask]
    # The data will be sorted anyway in np.unique and again in percentile, so we do it
    # here. Sorting also returns a contiguous array.
    col_data = np.sort(col_data)
    distinct_values = np.unique(col_data).astype(X_DTYPE)
    if len(distinct_values) <= max_bins:
        midpoints = distinct_values[:-1] + distinct_values[1:]
        midpoints *= 0.5
    else:
        # We could compute approximate midpoint percentiles using the output of
        # np.unique(col_data, return_counts) instead but this is more
        # work and the performance benefit will be limited because we
        # work on a fixed-size subsample of the full data.
        percentiles = np.linspace(0, 100, num=max_bins + 1)
        percentiles = percentiles[1:-1]
        midpoints = percentile(col_data, percentiles, method="midpoint").astype(X_DTYPE)
        assert midpoints.shape[0] == max_bins - 1

    # We avoid having +inf thresholds: +inf thresholds are only allowed in
    # a "split on nan" situation.
    np.clip(midpoints, a_min=None, a_max=ALMOST_INF, out=midpoints)
    return midpoints