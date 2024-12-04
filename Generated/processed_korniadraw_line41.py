import torch

def draw_line(image, p1, p2, color):
    """
    Draws a single line into an image.

    Parameters:
    - image (torch.Tensor): The input image with shape (C, H, W).
    - p1 (torch.Tensor): The start point [x, y] of the line with shape (2,) or (B, 2).
    - p2 (torch.Tensor): The end point [x, y] of the line with shape (2,) or (B, 2).
    - color (torch.Tensor): The color of the line with shape (C).

    Returns:
    - torch.Tensor: The image with the drawn line.
    """
    assert image.dim() == 3, "Image must have 3 dimensions (C, H, W)"
    C, H, W = image.shape
    assert color.shape == (C,), "Color must have the same number of channels as the image"
    
    if p1.dim() == 1:
        p1 = p1.unsqueeze(0)
    if p2.dim() == 1:
        p2 = p2.unsqueeze(0)
    
    assert p1.shape == p2.shape, "p1 and p2 must have the same shape"
    assert p1.shape[1] == 2, "p1 and p2 must have shape (2,) or (B, 2)"
    
    def bresenham(x0, y0, x1, y1):
        """Bresenham's line algorithm to get the coordinates of the line."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    for i in range(p1.shape[0]):
        x0, y0 = p1[i].tolist()
        x1, y1 = p2[i].tolist()
        
        # Ensure points are within bounds
        x0, y0 = max(0, min(W-1, x0)), max(0, min(H-1, y0))
        x1, y1 = max(0, min(W-1, x1)), max(0, min(H-1, y1))
        
        line_points = bresenham(x0, y0, x1, y1)
        
        for x, y in line_points:
            if 0 <= x < W and 0 <= y < H:
                image[:, y, x] = color

    return image

