import numpy as np
from scipy.stats import chi2 as chi2_dist
from sklearn.preprocessing import LabelBinarizer

def chi2(X, y):
    # Convert y to a binary matrix representation
    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    
    if Y.shape[1] == 1:
        Y = np.hstack([1 - Y, Y])
    
    # Number of samples and features
    n_samples, n_features = X.shape
    n_classes = Y.shape[1]
    
    # Initialize arrays to store chi2 statistics and p-values
    chi2_stats = np.zeros(n_features)
    p_values = np.zeros(n_features)
    
    # Calculate observed and expected frequencies
    observed = np.dot(Y.T, X)
    feature_sums = np.sum(X, axis=0)
    class_sums = np.sum(Y, axis=0).reshape(-1, 1)
    expected = np.dot(class_sums, feature_sums.reshape(1, -1)) / n_samples
    
    # Compute chi2 statistics and p-values for each feature
    chi2_stats = np.sum((observed - expected) ** 2 / expected, axis=0)
    p_values = chi2_dist.sf(chi2_stats, df=n_classes - 1)
    
    return chi2_stats, p_values

