def safe_diff(u, t, order=1):
    r"""The derivative of a variable with respect to another.
    Both tensors must have a shape of (n_samples, 1)
    See `this issue comment <https://github.com/NeuroDiffGym/neurodiffeq/issues/63#issuecomment-718007133>`_ for details

    :param u: The :math:`u` in :math:`\displaystyle\frac{\partial u}{\partial t}`.
    :type u: `torch.Tensor`
    :param t: The :math:`t` in :math:`\displaystyle\frac{\partial u}{\partial t}`.
    :type t: `torch.Tensor`
    :param order: The order of the derivative, defaults to 1.
    :type order: int
    :returns: The derivative evaluated at ``t``.
    :rtype: `torch.Tensor`
    """
    if len(u.shape) != 2 or len(t.shape) != 2 or u.shape[1] != 1 or t.shape[1] != 1:
        raise ValueError(f"Input shapes must both be (n_samples, 1) starting from neurodiffeq v0.2.0; \n"
                         f"got {u.shape} (for dependent variable) and {t.shape} (for independent variable)"
                         f"In most scenarios, consider reshaping inputs by `x = x.view(-1, 1)`\n"
                         f"For legacy usage, try `from neurodiffeq.neurodiffeq import unsafe_diff as diff`")
    if u.shape != t.shape:
        raise ValueError(f"Input shapes must be the same shape starting from v0.2.0; got {u.shape} != {t.shape}"
                         f"For legacy usage, try `from neurodiffeq.neurodiffeq import unsafe_diff as diff`")
    return unsafe_diff(u, t, order=order)