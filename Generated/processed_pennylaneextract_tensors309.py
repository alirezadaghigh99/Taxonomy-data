import pennylane as qml
from pennylane import numpy as np

def extract_tensors(x):
    """
    Iterate through an iterable, and extract any PennyLane tensors that appear.

    Args:
        x (.tensor or Sequence): an input tensor or sequence

    Yields:
        tensor: the next tensor in the sequence. If the input was a single
        tensor, then the tensor is yielded and the iterator completes.
    """
    if isinstance(x, (qml.numpy.tensor, np.ndarray)):
        yield x
    elif isinstance(x, (list, tuple, set)):
        for item in x:
            yield from extract_tensors(item)
    elif isinstance(x, dict):
        for key, value in x.items():
            yield from extract_tensors(value)

