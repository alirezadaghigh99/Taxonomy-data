import torch
from PIL import Image, ImageDraw, ImageFont
import warnings

def draw_bounding_boxes(image, boxes, labels=None, colors=None, fill=False, box_width=1, font=None, font_size=10):
    # Error handling
    if not isinstance(image, torch.Tensor):
        raise TypeError("The input image must be a PyTorch tensor.")
    
    if image.dtype not in [torch.uint8, torch.float32, torch.float64]:
        raise ValueError("The image dtype must be uint8 or float.")
    
    if len(image.shape) != 3 or image.shape[0] not in [1, 3]:
        raise ValueError("The image must have shape (C, H, W) with C being 1 or 3.")
    
    if not isinstance(boxes, torch.Tensor) or boxes.shape[1] != 4:
        raise ValueError("Bounding boxes must be a tensor of shape (N, 4) in (xmin, ymin, xmax, ymax) format.")
    
    if labels is not None and len(labels) != boxes.shape[0]:
        warnings.warn("The number of labels does not match the number of bounding boxes.")
    
    if colors is None:
        colors = ["red"] * boxes.shape[0]
    elif isinstance(colors, str):
        colors = [colors] * boxes.shape[0]
    elif len(colors) != boxes.shape[0]:
        warnings.warn("The number of colors does not match the number of bounding boxes. Using default color 'red'.")
        colors = ["red"] * boxes.shape[0]
    
    # Convert image tensor to PIL Image
    if image.dtype == torch.float32 or image.dtype == torch.float64:
        image = (image * 255).byte()
    
    if image.shape[0] == 1:
        image = image.squeeze(0).repeat(3, 1, 1)
    
    image = image.permute(1, 2, 0).cpu().numpy()
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image, "RGBA" if fill else "RGB")
    
    # Load font
    if font is not None:
        try:
            font = ImageFont.truetype(font, font_size)
        except IOError:
            warnings.warn("Font file not found. Using default font.")
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    
    # Draw bounding boxes
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box.tolist()
        color = colors[i]
        
        if fill:
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, fill=color + (100,), width=box_width)
        else:
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=box_width)
        
        if labels is not None:
            text_size = draw.textsize(labels[i], font=font)
            text_location = [xmin, ymin - text_size[1]]
            draw.rectangle([text_location, [xmin + text_size[0], ymin]], fill=color)
            draw.text((xmin, ymin - text_size[1]), labels[i], fill="white", font=font)
    
    # Convert PIL Image back to tensor
    result_image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1)
    
    return result_image

