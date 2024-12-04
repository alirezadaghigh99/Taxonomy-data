import numpy as np
import pandas as pd
from typing import Union

LabelLike = Union[list, np.ndarray, pd.Series, pd.DataFrame]

def labels_to_array(y: LabelLike) -> np.ndarray:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("DataFrame must have exactly one column")
        y = y.iloc[:, 0]  # Convert single-column DataFrame to Series
    
    if isinstance(y, pd.Series):
        y = y.values  # Convert Series to NumPy array
    
    if isinstance(y, list):
        y = np.array(y)  # Convert list to NumPy array
    
    if isinstance(y, np.ndarray):
        if y.ndim != 1:
            raise ValueError("Input array must be 1-dimensional")
        return y
    
    raise ValueError("Input type not supported or cannot be converted to a 1D NumPy array")

