import torch

def convert_image_dtype(image: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert a tensor image to the given dtype and scale the values accordingly.
    
    Args:
        image (torch.Tensor): Image to be converted.
        dtype (torch.dtype): Desired data type of the output.
    
    Returns:
        torch.Tensor: Converted image.
    
    Raises:
        RuntimeError: When trying to cast torch.float32 to torch.int32 or torch.int64 as
                      well as for trying to cast torch.float64 to torch.int64.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a torch.Tensor")
    
    if image.dtype == dtype:
        return image
    
    if dtype in [torch.int32, torch.int64] and image.dtype in [torch.float32, torch.float64]:
        raise RuntimeError(f"Cannot cast {image.dtype} to {dtype} due to potential overflow errors.")
    
    if image.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        if dtype in [torch.float32, torch.float64]:
            # Convert integer to float
            image = image.to(dtype)
            image = image / 255.0 if image.dtype == torch.uint8 else image / image.max()
        elif dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            # Convert integer to another integer type
            image = image.to(dtype)
    elif image.dtype in [torch.float32, torch.float64]:
        if dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            # Convert float to integer
            image = image * 255.0 if dtype == torch.uint8 else image * image.max()
            image = image.to(dtype)
        elif dtype in [torch.float32, torch.float64]:
            # Convert float to another float type
            image = image.to(dtype)
    
    return image

