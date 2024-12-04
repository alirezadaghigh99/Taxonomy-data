import numpy as np

def allclose(a, b, rtol=1e-05, atol=1e-08, **kwargs):
    """Wrapper around np.allclose, allowing tensors ``a`` and ``b``
    to differ in type"""
    try:
        # Some frameworks may provide their own allclose implementation.
        # Try and use it if available.
        res = np.allclose(a, b, rtol=rtol, atol=atol, **kwargs)
    except (TypeError, AttributeError, ImportError, RuntimeError):
        # Otherwise, convert the input to NumPy arrays.
        try:
            a = np.asarray(a)
            b = np.asarray(b)
        except Exception as e:
            raise ValueError(f"Failed to convert inputs to NumPy arrays: {e}")
        
        res = np.allclose(a, b, rtol=rtol, atol=atol, **kwargs)
    
    return res