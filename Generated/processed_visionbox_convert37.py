import torch

def box_convert(boxes, in_fmt, out_fmt):
    """
    Convert boxes from in_fmt to out_fmt.

    Args:
        boxes (Tensor[N, 4]): boxes which will be converted.
        in_fmt (str): Input format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh'].
        out_fmt (str): Output format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh']

    Returns:
        Tensor[N, 4]: Boxes into converted format.
    """
    if in_fmt == out_fmt:
        return boxes

    if in_fmt == 'xyxy':
        x1, y1, x2, y2 = boxes.unbind(-1)
        if out_fmt == 'xywh':
            return torch.stack((x1, y1, x2 - x1, y2 - y1), dim=-1)
        elif out_fmt == 'cxcywh':
            return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)

    elif in_fmt == 'xywh':
        x1, y1, w, h = boxes.unbind(-1)
        if out_fmt == 'xyxy':
            return torch.stack((x1, y1, x1 + w, y1 + h), dim=-1)
        elif out_fmt == 'cxcywh':
            return torch.stack((x1 + w / 2, y1 + h / 2, w, h), dim=-1)

    elif in_fmt == 'cxcywh':
        cx, cy, w, h = boxes.unbind(-1)
        if out_fmt == 'xyxy':
            return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)
        elif out_fmt == 'xywh':
            return torch.stack((cx - w / 2, cy - h / 2, w, h), dim=-1)

    else:
        raise ValueError(f"Unsupported format: {in_fmt}")

    raise ValueError(f"Unsupported format: {out_fmt}")

