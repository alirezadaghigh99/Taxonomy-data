import torch
from torch.autograd import Function
from torch import Tensor
from typing import Any, Optional, Callable

class STEFunction(Function):
    """Straight-Through Estimation (STE) function.

    STE bridges the gradients between the input tensor and output tensor as if the function
    was an identity function. Meanwhile, advanced gradient functions are also supported. e.g.
    the output gradients can be mapped into [-1, 1] with ``F.hardtanh`` function.

    Args:
        grad_fn: function to restrain the gradient received. If None, no mapping will performed.

    Example:
        Let the gradients of ``torch.sign`` estimated from STE.
        >>> input = torch.randn(4, requires_grad = True)
        >>> output = torch.sign(input)
        >>> loss = output.mean()
        >>> loss.backward()
        >>> input.grad
        tensor([0., 0., 0., 0.])

        >>> with torch.no_grad():
        ...     output = torch.sign(input)
        >>> out_est = STEFunction.apply(input, output)
        >>> loss = out_est.mean()
        >>> loss.backward()
        >>> input.grad
        tensor([0.2500, 0.2500, 0.2500, 0.2500])
    """

    @staticmethod
    def forward(ctx: Any, input: Tensor, output: Tensor, grad_fn: Optional[Callable[..., Any]] = None) -> Tensor:
        # Save context for backward pass
        ctx.save_for_backward(input, output)
        ctx.grad_fn = grad_fn
        # Return the output tensor as is
        return output