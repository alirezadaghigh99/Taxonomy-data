import torch
import torch.nn.functional as F
import numpy as np

def canny(input, low_threshold, high_threshold, kernel_size, sigma, hysteresis=True, eps=1e-6):
    assert len(input.shape) == 4, "Input tensor must have shape (B, C, H, W)"
    B, C, H, W = input.shape
    
    # Convert to grayscale if input has 3 channels
    if C == 3:
        input = 0.299 * input[:, 0, :, :] + 0.587 * input[:, 1, :, :] + 0.114 * input[:, 2, :, :]
        input = input.unsqueeze(1)
    elif C == 1:
        input = input
    else:
        raise ValueError("Input tensor must have 1 or 3 channels")
    
    # Apply Gaussian blur
    def gaussian_kernel(kernel_size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1/ (2 * np.pi * sigma**2)) * np.exp(- ((x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2) / (2 * sigma**2)),
            (kernel_size, kernel_size)
        )
        return torch.tensor(kernel / np.sum(kernel), dtype=torch.float32)
    
    kernel = gaussian_kernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(C, 1, 1, 1)
    input = F.conv2d(input, kernel, padding=kernel_size//2, groups=C)
    
    # Compute gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    grad_x = F.conv2d(input, sobel_x, padding=1)
    grad_y = F.conv2d(input, sobel_y, padding=1)
    
    # Compute gradient magnitude and angle
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + eps)
    grad_angle = torch.atan2(grad_y, grad_x)
    
    # Non-maximal suppression
    def non_max_suppression(magnitude, angle):
        B, C, H, W = magnitude.shape
        output = torch.zeros_like(magnitude)
        angle = angle * 180.0 / np.pi
        angle[angle < 0] += 180
        
        for b in range(B):
            for i in range(1, H-1):
                for j in range(1, W-1):
                    q = 255
                    r = 255
                    # Angle 0
                    if (0 <= angle[b, 0, i, j] < 22.5) or (157.5 <= angle[b, 0, i, j] <= 180):
                        q = magnitude[b, 0, i, j+1]
                        r = magnitude[b, 0, i, j-1]
                    # Angle 45
                    elif 22.5 <= angle[b, 0, i, j] < 67.5:
                        q = magnitude[b, 0, i+1, j-1]
                        r = magnitude[b, 0, i-1, j+1]
                    # Angle 90
                    elif 67.5 <= angle[b, 0, i, j] < 112.5:
                        q = magnitude[b, 0, i+1, j]
                        r = magnitude[b, 0, i-1, j]
                    # Angle 135
                    elif 112.5 <= angle[b, 0, i, j] < 157.5:
                        q = magnitude[b, 0, i-1, j-1]
                        r = magnitude[b, 0, i+1, j+1]
                    
                    if (magnitude[b, 0, i, j] >= q) and (magnitude[b, 0, i, j] >= r):
                        output[b, 0, i, j] = magnitude[b, 0, i, j]
                    else:
                        output[b, 0, i, j] = 0
        return output
    
    suppressed = non_max_suppression(grad_magnitude, grad_angle)
    
    # Apply thresholding
    strong_edges = (suppressed >= high_threshold).float()
    weak_edges = ((suppressed >= low_threshold) & (suppressed < high_threshold)).float()
    
    if hysteresis:
        # Hysteresis edge tracking
        def hysteresis_tracking(strong, weak):
            B, C, H, W = strong.shape
            output = strong.clone()
            for b in range(B):
                for i in range(1, H-1):
                    for j in range(1, W-1):
                        if weak[b, 0, i, j] == 1:
                            if ((strong[b, 0, i+1, j-1:j+2] == 1).any() or
                                (strong[b, 0, i-1, j-1:j+2] == 1).any() or
                                (strong[b, 0, i, [j-1, j+1]] == 1).any()):
                                output[b, 0, i, j] = 1
            return output
        
        edges = hysteresis_tracking(strong_edges, weak_edges)
    else:
        edges = strong_edges
    
    return grad_magnitude, edges

