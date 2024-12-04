from typing import Any, Callable, Optional, Tuple
import torch
from torch import Tensor
from torch.autograd import Function

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
        ctx.in_shape = input.shape
        ctx.out_shape = output.shape
        ctx.grad_fn = grad_fn
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, Tensor, None]:
        # Apply the gradient function if provided
        if ctx.grad_fn is not None:
            grad_input = ctx.grad_fn(grad_output)
        else:
            grad_input = grad_output

        # Return the gradient for the input, None for the output, and None for grad_fn
        return grad_input, None, None