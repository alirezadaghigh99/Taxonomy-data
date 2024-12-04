import numpy as np

def softmax(x: np.ndarray, temperature: float = 1.0, axis: int = None, shift: bool = True) -> np.ndarray:
    """
    Apply the softmax function to the input array.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    temperature : float
        Temperature of the softmax function.
    axis : Optional[int]
        Axis to apply the softmax function. If None, the softmax function is
        applied to all elements of the input array.
    shift : bool
        Whether to shift the input array before applying the softmax function.
        This is useful to avoid numerical issues when the input array contains
        large values, that could result in overflows when applying the exponential
        function.

    Returns
    -------
    np.ndarray
        Softmax function applied to the input array.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    # Adjust the input array by the temperature
    x = x / temperature

    if shift:
        # Shift the input array to avoid numerical issues
        x_max = np.max(x, axis=axis, keepdims=True)
        x = x - x_max

    # Compute the exponentials
    exp_x = np.exp(x)

    # Compute the sum of exponentials along the specified axis
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)

    # Compute the softmax values
    softmax_values = exp_x / sum_exp_x

    return softmax_values

