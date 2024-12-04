import torch
from PIL import Image, ImageDraw

def draw_keypoints(image, keypoints, connectivity=None, colors='red', radius=3, width=2, visibility=None):
    # Validate input image tensor
    if not isinstance(image, torch.Tensor):
        raise TypeError("The input image must be a tensor.")
    
    if image.dtype not in [torch.uint8, torch.float]:
        raise ValueError("The image dtype must be uint8 or float.")
    
    if image.ndimension() != 3 or image.shape[0] != 3:
        raise ValueError("The image must have shape (3, H, W).")
    
    # Validate keypoints tensor
    if not isinstance(keypoints, torch.Tensor):
        raise TypeError("The keypoints must be a tensor.")
    
    if keypoints.ndimension() != 3 or keypoints.shape[2] != 2:
        raise ValueError("The keypoints tensor must have shape (num_instances, K, 2).")
    
    # Validate visibility tensor
    if visibility is not None:
        if not isinstance(visibility, torch.Tensor):
            raise TypeError("The visibility must be a tensor.")
        
        if visibility.shape != keypoints.shape[:2]:
            raise ValueError("The visibility tensor must match the shape of the keypoints tensor.")
    
    # Convert image tensor to PIL Image
    if image.dtype == torch.float:
        image = (image * 255).byte()
    image = image.permute(1, 2, 0).cpu().numpy()
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Draw keypoints and connections
    num_instances, K, _ = keypoints.shape
    for i in range(num_instances):
        for j in range(K):
            if visibility is None or visibility[i, j]:
                x, y = keypoints[i, j].tolist()
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colors)
        
        if connectivity:
            for (start, end) in connectivity:
                if (visibility is None or (visibility[i, start] and visibility[i, end])):
                    x1, y1 = keypoints[i, start].tolist()
                    x2, y2 = keypoints[i, end].tolist()
                    draw.line((x1, y1, x2, y2), fill=colors, width=width)
    
    # Convert PIL Image back to tensor
    result_image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1)
    
    return result_image

