from PIL import Image
import io
import torch

def encode_jpeg(images, quality):
    """
    Encodes an input tensor or list of tensors representing images into JPEG format.
    
    Args:
        images (torch.Tensor or list of torch.Tensor): Input images in CHW layout.
        quality (int): Quality parameter between 1 and 100 to control the output JPEG quality.
        
    Returns:
        torch.Tensor or list of torch.Tensor: Tensor or list of tensors containing the raw bytes of the JPEG file(s).
        
    Raises:
        ValueError: If the quality is not between 1 and 100 or if an empty list is passed as input.
    """
    if not (1 <= quality <= 100):
        raise ValueError("Quality must be between 1 and 100.")
    
    if isinstance(images, list) and len(images) == 0:
        raise ValueError("Input list of images cannot be empty.")
    
    def encode_single_image(image):
        if not isinstance(image, torch.Tensor):
            raise TypeError("Each image must be a torch.Tensor.")
        
        if image.ndimension() != 3 or image.size(0) != 3:
            raise ValueError("Each image must be in CHW layout with 3 channels.")
        
        # Convert CHW to HWC
        image = image.permute(1, 2, 0).contiguous()
        
        # Convert to numpy array
        image_np = image.numpy()
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_np.astype('uint8'), 'RGB')
        
        # Encode to JPEG
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        jpeg_bytes = buffer.getvalue()
        
        # Convert to torch tensor
        jpeg_tensor = torch.tensor(list(jpeg_bytes), dtype=torch.uint8)
        
        return jpeg_tensor
    
    if isinstance(images, torch.Tensor):
        return encode_single_image(images)
    elif isinstance(images, list):
        return [encode_single_image(image) for image in images]
    else:
        raise TypeError("Input must be a torch.Tensor or a list of torch.Tensor.")

