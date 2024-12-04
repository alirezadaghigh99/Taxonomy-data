from pathlib import Path
import torch
import kornia as K
import kornia.augmentation as Kaug
import kornia.io as Kio

def _load_image_to_tensor(path_file: Path, device: torch.device) -> torch.Tensor:
    # Check if the file exists
    if not path_file.exists():
        raise FileNotFoundError(f"The file {path_file} does not exist.")
    
    # Check if the file extension is supported
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    if path_file.suffix.lower() not in supported_formats:
        raise ValueError(f"Unsupported image format: {path_file.suffix}. Supported formats are: {supported_formats}")
    
    # Load and decode the image using Kornia
    try:
        image = Kio.load_image(str(path_file), Kio.ImageLoadType.RGB32)
    except Exception as e:
        raise RuntimeError(f"Failed to load and decode the image: {e}")
    
    # Convert the image to a PyTorch tensor and move it to the specified device
    image_tensor = image.to(device)
    
    # Ensure the tensor has the shape (3, H, W)
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError(f"Unexpected image tensor shape: {image_tensor.shape}. Expected shape is (3, H, W).")
    
    return image_tensor

