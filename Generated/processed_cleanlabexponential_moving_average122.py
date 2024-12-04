import numpy as np

def exponential_moving_average(s, alpha=0.5, axis=0, **kwargs):
    """
    Calculate the exponential moving average (EMA) of the given array of scores.

    Parameters:
    s (np.ndarray): Array of scores.
    alpha (float): The forgetting factor that determines the weight of the previous EMA score.
    axis (int): The axis along which the scores are sorted.
    **kwargs: Additional keyword arguments.

    Returns:
    np.ndarray: The exponential moving average score.
    """
    s = np.asarray(s)
    
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be between 0 and 1.")
    
    if axis < 0:
        axis += s.ndim
    
    if axis >= s.ndim or axis < 0:
        raise ValueError("axis must be within the range of input array dimensions.")
    
    # Move the specified axis to the first dimension
    s = np.moveaxis(s, axis, 0)
    
    # Initialize the EMA array
    s_ema = np.zeros_like(s)
    
    # Set the first value of EMA to the first value of the scores
    s_ema[0] = s[0]
    
    # Calculate the EMA for the rest of the values
    for t in range(1, s.shape[0]):
        s_ema[t] = alpha * s[t] + (1 - alpha) * s_ema[t - 1]
    
    # Move the axis back to its original position
    s_ema = np.moveaxis(s_ema, 0, axis)
    
    return s_ema

