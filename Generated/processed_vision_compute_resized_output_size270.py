def _compute_resized_output_size(image_size, size=None, max_size=None, allow_size_none=False):
    """
    Calculate the new size of an image after resizing.

    Parameters:
    - image_size (tuple): Original size of the image (height, width).
    - size (list or tuple or int, optional): Desired size of the smaller edge or both dimensions.
    - max_size (int, optional): Limits the size of the longer edge.
    - allow_size_none (bool): Permits `size` to be `None`.

    Returns:
    - list: New size [new_height, new_width].

    Raises:
    - ValueError: If `size` is `None` but `max_size` is not an integer, if `max_size` is smaller than the requested size, or if any other invalid configuration is encountered.
    """
    if size is None:
        if not allow_size_none:
            raise ValueError("`size` cannot be None unless `allow_size_none` is True.")
        if max_size is None or not isinstance(max_size, int):
            raise ValueError("`max_size` must be an integer when `size` is None.")
        # If size is None and max_size is provided, we assume max_size is the target for the longer edge
        original_height, original_width = image_size
        if original_height > original_width:
            new_height = max_size
            new_width = int(max_size * original_width / original_height)
        else:
            new_width = max_size
            new_height = int(max_size * original_height / original_width)
        return [new_height, new_width]

    if isinstance(size, int):
        size = [size]

    if isinstance(size, (list, tuple)):
        if len(size) == 1:
            size = size[0]
            original_height, original_width = image_size
            if original_height < original_width:
                new_height = size
                new_width = int(size * original_width / original_height)
            else:
                new_width = size
                new_height = int(size * original_height / original_width)
        elif len(size) == 2:
            new_height, new_width = size
        else:
            raise ValueError("`size` must be an int, a list/tuple of one or two elements, or None.")
    else:
        raise ValueError("`size` must be an int, a list/tuple of one or two elements, or None.")

    if max_size is not None:
        if not isinstance(max_size, int):
            raise ValueError("`max_size` must be an integer.")
        if max(new_height, new_width) > max_size:
            if new_height > new_width:
                new_height = max_size
                new_width = int(max_size * new_width / new_height)
            else:
                new_width = max_size
                new_height = int(max_size * new_height / new_width)

    return [new_height, new_width]

