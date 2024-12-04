from collections import Counter

def ngrams(sequence, n):
    """
    Generate the n-grams from a sequence of items.

    Args:
        sequence: sequence of items
        n: n-gram order

    Returns:
        A counter of ngram objects
    """
    # Ensure the sequence is long enough to form n-grams
    if len(sequence) < n:
        return Counter()
    
    # Generate the n-grams
    ngram_list = [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]
    
    # Count the occurrences of each n-gram
    ngram_counter = Counter(ngram_list)
    
    return ngram_counter

