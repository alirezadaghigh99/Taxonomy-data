import torch
import torch.nn.functional as F

def make_grid(tensor, nrow=8, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0):
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)
    
    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # single-channel image
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)
    
    if normalize:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(value_range, (tuple, list)) and len(value_range) == 2, \
                "value_range should be a tuple (min, max)"
        else:
            value_range = (tensor.min(), tensor.max())
        
        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)
        
        def norm_range(t, value_range):
            if scale_each:
                for t_img in t:
                    norm_ip(t_img, value_range[0], value_range[1])
            else:
                norm_ip(t, value_range[0], value_range[1])
        
        norm_range(tensor, value_range)
    
    if tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)
    
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding).narrow(2, x * width + padding, width - padding).copy_(tensor[k])
            k = k + 1
    
    return grid

