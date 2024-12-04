def convert_bounding_box_format(
    inpt: torch.Tensor,
    old_format: Optional[BoundingBoxFormat] = None,
    new_format: Optional[BoundingBoxFormat] = None,
    inplace: bool = False,
) -> torch.Tensor:
    """See :func:`~torchvision.transforms.v2.ConvertBoundingBoxFormat` for details."""
    # This being a kernel / functional hybrid, we need an option to pass `old_format` explicitly for pure tensor
    # inputs as well as extract it from `tv_tensors.BoundingBoxes` inputs. However, putting a default value on
    # `old_format` means we also need to put one on `new_format` to have syntactically correct Python. Here we mimic the
    # default error that would be thrown if `new_format` had no default value.
    if new_format is None:
        raise TypeError("convert_bounding_box_format() missing 1 required argument: 'new_format'")

    if not torch.jit.is_scripting():
        _log_api_usage_once(convert_bounding_box_format)

    if isinstance(old_format, str):
        old_format = BoundingBoxFormat[old_format.upper()]
    if isinstance(new_format, str):
        new_format = BoundingBoxFormat[new_format.upper()]

    if torch.jit.is_scripting() or is_pure_tensor(inpt):
        if old_format is None:
            raise ValueError("For pure tensor inputs, `old_format` has to be passed.")
        return _convert_bounding_box_format(inpt, old_format=old_format, new_format=new_format, inplace=inplace)
    elif isinstance(inpt, tv_tensors.BoundingBoxes):
        if old_format is not None:
            raise ValueError("For bounding box tv_tensor inputs, `old_format` must not be passed.")
        output = _convert_bounding_box_format(
            inpt.as_subclass(torch.Tensor), old_format=inpt.format, new_format=new_format, inplace=inplace
        )
        return tv_tensors.wrap(output, like=inpt, format=new_format)
    else:
        raise TypeError(
            f"Input can either be a plain tensor or a bounding box tv_tensor, but got {type(inpt)} instead."
        )