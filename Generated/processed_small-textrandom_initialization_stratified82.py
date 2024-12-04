import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

def stratified_sampling(y, n_samples):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_ratios = class_counts / len(y)
    n_samples_per_class = (class_ratios * n_samples).astype(int)
    
    indices = []
    for cls, n_cls_samples in zip(unique_classes, n_samples_per_class):
        cls_indices = np.where(y == cls)[0]
        sampled_indices = np.random.choice(cls_indices, n_cls_samples, replace=False)
        indices.extend(sampled_indices)
    
    return np.array(indices)

def multilabel_stratified_subsets_sampling(y, n_samples):
    # This is a placeholder for the actual implementation of multilabel stratified sampling
    # For simplicity, we will use a similar approach to stratified_sampling but for labelsets
    labelsets, labelset_counts = np.unique(y.toarray(), axis=0, return_counts=True)
    labelset_ratios = labelset_counts / len(y)
    n_samples_per_labelset = (labelset_ratios * n_samples).astype(int)
    
    indices = []
    for labelset, n_labelset_samples in zip(labelsets, n_samples_per_labelset):
        labelset_indices = np.where((y.toarray() == labelset).all(axis=1))[0]
        sampled_indices = np.random.choice(labelset_indices, n_labelset_samples, replace=False)
        indices.extend(sampled_indices)
    
    return np.array(indices)

def random_initialization_stratified(y, n_samples=10, multilabel_strategy='labelsets'):
    if isinstance(y, csr_matrix):
        if multilabel_strategy == 'labelsets':
            return multilabel_stratified_subsets_sampling(y, n_samples)
        else:
            raise ValueError(f'Invalid multilabel_strategy: {multilabel_strategy}')
    elif isinstance(y, np.ndarray):
        return stratified_sampling(y, n_samples)
    else:
        raise TypeError('y must be either an np.ndarray or csr_matrix')

