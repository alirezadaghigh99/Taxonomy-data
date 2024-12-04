import torch
import numpy as np

def flow_to_image(flow):
    if not isinstance(flow, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    
    if flow.dtype != torch.float:
        raise ValueError("Input tensor must be of type torch.float")
    
    if flow.dim() not in [3, 4] or (flow.dim() == 3 and flow.size(0) != 2) or (flow.dim() == 4 and flow.size(1) != 2):
        raise ValueError("Input tensor must have shape (2, H, W) or (N, 2, H, W)")
    
    def normalize_flow(flow):
        max_flow = torch.max(torch.abs(flow))
        return flow / (max_flow + 1e-5)
    
    def flow_to_rgb(flow):
        H, W = flow.shape[1], flow.shape[2]
        rgb = torch.zeros(3, H, W, dtype=torch.float)
        
        u = flow[0]
        v = flow[1]
        
        rad = torch.sqrt(u**2 + v**2)
        a = torch.atan2(-v, -u) / np.pi
        
        fk = (a + 1) / 2 * (255 - 1)
        k0 = torch.floor(fk).long()
        k1 = k0 + 1
        
        f = fk - k0
        
        colorwheel = make_colorwheel()
        ncols = colorwheel.shape[0]
        
        for i in range(3):
            col0 = colorwheel[k0 % ncols, i] / 255.0
            col1 = colorwheel[k1 % ncols, i] / 255.0
            col = (1 - f) * col0 + f * col1
            
            col = 1 - rad * (1 - col)
            col[rad <= 1] = 1 - rad[rad <= 1] * (1 - col[rad <= 1])
            col[rad > 1] *= 0.75
            
            rgb[i] = col
        
        return rgb
    
    def make_colorwheel():
        # Create a color wheel for visualization
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6
        
        ncols = RY + YG + GC + CB + BM + MR
        colorwheel = torch.zeros((ncols, 3))
        
        col = 0
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY) / RY)
        col += RY
        
        colorwheel[col:col+YG, 0] = 255 - torch.floor(255 * torch.arange(0, YG) / YG)
        colorwheel[col:col+YG, 1] = 255
        col += YG
        
        colorwheel[col:col+GC, 1] = 255
        colorwheel[col:col+GC, 2] = torch.floor(255 * torch.arange(0, GC) / GC)
        col += GC
        
        colorwheel[col:col+CB, 1] = 255 - torch.floor(255 * torch.arange(0, CB) / CB)
        colorwheel[col:col+CB, 2] = 255
        col += CB
        
        colorwheel[col:col+BM, 2] = 255
        colorwheel[col:col+BM, 0] = torch.floor(255 * torch.arange(0, BM) / BM)
        col += BM
        
        colorwheel[col:col+MR, 2] = 255 - torch.floor(255 * torch.arange(0, MR) / MR)
        colorwheel[col:col+MR, 0] = 255
        
        return colorwheel
    
    if flow.dim() == 3:
        flow = normalize_flow(flow)
        rgb_image = flow_to_rgb(flow)
    else:
        N = flow.size(0)
        rgb_image = torch.zeros(N, 3, flow.size(2), flow.size(3), dtype=torch.float)
        for i in range(N):
            normalized_flow = normalize_flow(flow[i])
            rgb_image[i] = flow_to_rgb(normalized_flow)
    
    return rgb_image

