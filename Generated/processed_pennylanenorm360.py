def norm(tensor, interface="scipy", axis=None, **kwargs):
    if interface == "jax":
        import jax.numpy as jnp
        return jnp.linalg.norm(tensor, axis=axis, **kwargs)
    
    elif interface == "tensorflow":
        import tensorflow as tf
        return tf.norm(tensor, axis=axis, **kwargs)
    
    elif interface == "torch":
        import torch
        if axis is not None:
            if isinstance(axis, int):
                axis = (axis,)
            return torch.linalg.norm(tensor, dim=axis, **kwargs)
        else:
            return torch.linalg.norm(tensor, **kwargs)
    
    elif interface == "autograd":
        import autograd.numpy as anp
        def _flat_autograd_norm(x):
            return anp.sqrt(anp.sum(x**2))
        
        if kwargs.get('use_flat', False):
            return _flat_autograd_norm(tensor)
        else:
            return anp.linalg.norm(tensor, axis=axis, **kwargs)
    
    else:  # default to scipy
        import numpy as np
        return np.linalg.norm(tensor, axis=axis, **kwargs)

