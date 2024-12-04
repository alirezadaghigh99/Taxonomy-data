import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import _num_samples
from sklearn.model_selection import StratifiedShuffleSplit

def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    """
    Split arrays or matrices into random train and test subsets.
    
    Parameters:
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
    test_size : float, int, or None, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
        If None, the value is set to the complement of the train size.
    train_size : float, int, or None, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
        If int, represents the absolute number of train samples.
        If None, the value is automatically set to the complement of the test size.
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting.
    stratify : array-like or None, default=None
        If not None, data is split in a stratified fashion, using this as the class labels.
    
    Returns:
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
    """
    
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    
    # Validate input arrays
    for array in arrays:
        if _num_samples(array) != _num_samples(arrays[0]):
            raise ValueError("All input arrays must have the same number of samples")
    
    n_samples = _num_samples(arrays[0])
    
    if test_size is None and train_size is None:
        test_size = 0.25
    
    if test_size is not None and isinstance(test_size, float):
        if test_size < 0 or test_size > 1:
            raise ValueError("test_size should be between 0.0 and 1.0")
        test_size = int(np.ceil(test_size * n_samples))
    
    if train_size is not None and isinstance(train_size, float):
        if train_size < 0 or train_size > 1:
            raise ValueError("train_size should be between 0.0 and 1.0")
        train_size = int(np.floor(train_size * n_samples))
    
    if test_size is None:
        test_size = n_samples - train_size
    
    if train_size is None:
        train_size = n_samples - test_size
    
    if train_size + test_size > n_samples:
        raise ValueError("The sum of train_size and test_size should be smaller than the number of samples")
    
    if stratify is not None:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, train_size=train_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(arrays[0], stratify))
    else:
        if shuffle:
            rng = check_random_state(random_state)
            permutation = rng.permutation(n_samples)
        else:
            permutation = np.arange(n_samples)
        
        test_idx = permutation[:test_size]
        train_idx = permutation[test_size:test_size + train_size]
    
    result = []
    for array in arrays:
        result.append(array[train_idx])
        result.append(array[test_idx])
    
    return result

