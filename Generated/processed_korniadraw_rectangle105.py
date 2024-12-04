import torch

def draw_rectangle(image, rectangle, color=None, fill=False):
    """
    Draws one or more rectangles on a batch of image tensors.

    Parameters:
    - image: A tensor of shape (B, C, H, W).
    - rectangle: A tensor of shape (B, N, 4).
    - color: An optional tensor specifying the color of the rectangles.
    - fill: An optional boolean flag indicating whether to fill the rectangles with color (True) or just draw the borders (False). Defaults to False.

    Returns:
    - The modified image tensor.
    """
    B, C, H, W = image.shape
    B_rect, N, points = rectangle.shape

    # Error handling
    assert B == B_rect, "Batch size of image tensor and rectangle tensor must match."
    assert points == 4, "Each rectangle must be defined by four coordinates (x1, y1, x2, y2)."

    # Default color is white if not provided
    if color is None:
        color = torch.tensor([255] * C, dtype=image.dtype, device=image.device)
    else:
        color = color.to(image.dtype).to(image.device)

    # Broadcast color to the appropriate shape
    if color.dim() == 1:
        color = color.view(1, 1, -1)
    elif color.dim() == 2:
        color = color.view(B, N, -1)
    elif color.dim() == 3:
        color = color.view(B, N, C)

    for b in range(B):
        for n in range(N):
            x1, y1, x2, y2 = rectangle[b, n]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if fill:
                image[b, :, y1:y2, x1:x2] = color[b, n].view(C, 1, 1)
            else:
                # Draw borders
                image[b, :, y1, x1:x2] = color[b, n].view(C)
                image[b, :, y2-1, x1:x2] = color[b, n].view(C)
                image[b, :, y1:y2, x1] = color[b, n].view(C).view(C, 1)
                image[b, :, y1:y2, x2-1] = color[b, n].view(C).view(C, 1)

    return image

