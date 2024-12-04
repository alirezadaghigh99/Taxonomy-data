from collections import Counter
from nltk.util import ngrams

def get_ngrams(sequence, n):
    """Generate n-grams from a sequence of items."""
    return list(ngrams(sequence, n))

def modified_precision(references, candidate, n):
    """
    Calculate the modified precision for a given list of references, a candidate translation, and an n-gram order.
    
    Args:
    references (list of list of str): List of reference translations.
    candidate (list of str): Candidate translation.
    n (int): The n-gram order.
    
    Returns:
    tuple: (sum of clipped counts, sum of candidate counts)
    """
    # Generate n-grams for the candidate translation
    candidate_ngrams = get_ngrams(candidate, n)
    candidate_counts = Counter(candidate_ngrams)
    
    # Generate n-grams for each reference translation
    reference_counts = Counter()
    for reference in references:
        reference_ngrams = get_ngrams(reference, n)
        reference_counts |= Counter(reference_ngrams)  # Union of counts to get max counts
    
    # Calculate clipped counts
    clipped_counts = {ngram: min(count, reference_counts[ngram]) for ngram, count in candidate_counts.items()}
    
    # Sum of clipped counts and total candidate counts
    sum_clipped_counts = sum(clipped_counts.values())
    sum_candidate_counts = sum(candidate_counts.values())
    
    return sum_clipped_counts, sum_candidate_counts

