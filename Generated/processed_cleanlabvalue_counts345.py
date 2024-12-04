import numpy as np
from collections import Counter

def value_counts(x, num_classes=None, multi_label=False):
    if multi_label:
        # Flatten the list of iterables
        x = [item for sublist in x for item in sublist]
    
    # Count the occurrences of each unique item
    counter = Counter(x)
    
    if num_classes is None:
        # If num_classes is not provided, use the unique items in x
        unique_items = sorted(counter.keys())
    else:
        # If num_classes is provided, use the range of num_classes
        unique_items = range(num_classes)
    
    # Create an array to store the counts
    counts = np.zeros((len(unique_items), 1), dtype=int)
    
    for i, item in enumerate(unique_items):
        counts[i] = counter.get(item, 0)
    
    return counts

