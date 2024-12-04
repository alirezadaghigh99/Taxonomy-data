import torch
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
from torchvision.transforms import InterpolationMode

def perspective(img, startpoints, endpoints, interpolation=InterpolationMode.BILINEAR, fill=0):
    if isinstance(img, Image.Image):
        return _perspective_pil(img, startpoints, endpoints, interpolation, fill)
    elif isinstance(img, torch.Tensor):
        return _perspective_tensor(img, startpoints, endpoints, interpolation, fill)
    else:
        raise TypeError("img should be PIL Image or Tensor")

def _perspective_pil(img, startpoints, endpoints, interpolation, fill):
    width, height = img.size
    coeffs = _find_coeffs(endpoints, startpoints)
    return img.transform((width, height), Image.PERSPECTIVE, coeffs, interpolation, fillcolor=fill)

def _perspective_tensor(img, startpoints, endpoints, interpolation, fill):
    if img.ndim < 2:
        raise ValueError("Tensor should have at least 2 dimensions")
    
    if img.ndim == 2:
        img = img.unsqueeze(0)
    
    b, c, h, w = img.shape[-4:]
    dtype = img.dtype
    device = img.device
    
    startpoints = torch.tensor(startpoints, dtype=dtype, device=device)
    endpoints = torch.tensor(endpoints, dtype=dtype, device=device)
    
    matrix = _get_perspective_transform(startpoints, endpoints)
    grid = F.affine_grid(matrix.unsqueeze(0), img.unsqueeze(0).size(), align_corners=False)
    img = F.grid_sample(img.unsqueeze(0), grid, mode=interpolation.value, padding_mode='zeros', align_corners=False)
    
    if fill != 0:
        mask = F.grid_sample(torch.ones_like(img), grid, mode='nearest', padding_mode='zeros', align_corners=False)
        img = img * mask + (1 - mask) * fill
    
    return img.squeeze(0)

def _find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    
    A = np.matrix(matrix, dtype=np.float32)
    B = np.array(pb).reshape(8)
    
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def _get_perspective_transform(startpoints, endpoints):
    startpoints = startpoints.float()
    endpoints = endpoints.float()
    
    A = []
    for i in range(4):
        x, y = startpoints[i]
        u, v = endpoints[i]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
    
    A = torch.tensor(A, dtype=torch.float32)
    B = endpoints.view(8)
    
    res, _ = torch.solve(B.unsqueeze(1), A)
    return res.view(3, 3)

