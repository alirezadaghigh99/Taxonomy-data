import torch
import torch.nn.functional as F

def equalize_clahe(input, clip_limit, grid_size, slow_and_differentiable):
    # Input validation
    if not isinstance(clip_limit, float):
        raise TypeError("clip_limit must be a float")
    if not (isinstance(grid_size, tuple) and len(grid_size) == 2 and all(isinstance(x, int) for x in grid_size)):
        raise TypeError("grid_size must be a tuple of two integers")
    if any(x <= 0 for x in grid_size):
        raise ValueError("All elements of grid_size must be positive")

    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")

    # Ensure input values are in the range [0, 1]
    if input.min() < 0 or input.max() > 1:
        raise ValueError("input values must be in the range [0, 1]")

    # Get the shape of the input tensor
    original_shape = input.shape
    batch_dims = original_shape[:-3]
    C, H, W = original_shape[-3:]

    # Reshape input to (N, C, H, W) where N is the product of batch dimensions
    input = input.view(-1, C, H, W)
    N = input.shape[0]

    # Calculate tile size
    tile_h, tile_w = H // grid_size[0], W // grid_size[1]

    # Initialize output tensor
    output = torch.zeros_like(input)

    # Process each image in the batch
    for n in range(N):
        for c in range(C):
            img = input[n, c]

            # Create an empty array to store the CLAHE result
            clahe_img = torch.zeros_like(img)

            # Process each tile
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    # Define the tile region
                    y1, y2 = i * tile_h, (i + 1) * tile_h
                    x1, x2 = j * tile_w, (j + 1) * tile_w

                    # Extract the tile
                    tile = img[y1:y2, x1:x2]

                    # Compute the histogram
                    hist = torch.histc(tile, bins=256, min=0, max=1)

                    # Clip the histogram if clip_limit is set
                    if clip_limit > 0:
                        excess = hist - clip_limit
                        excess[excess < 0] = 0
                        hist = hist + excess.sum() / 256

                    # Compute the cumulative distribution function (CDF)
                    cdf = hist.cumsum(0)
                    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
                    cdf = cdf * 255

                    # Map the tile pixels using the CDF
                    tile_flat = (tile * 255).long().view(-1)
                    clahe_tile = cdf[tile_flat].view(tile.shape) / 255

                    # Place the CLAHE tile back into the image
                    clahe_img[y1:y2, x1:x2] = clahe_tile

            # Interpolate between tiles
            if slow_and_differentiable:
                # Use bilinear interpolation for smooth transitions
                clahe_img = F.interpolate(clahe_img.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze()
            else:
                # Use nearest neighbor interpolation for faster processing
                clahe_img = F.interpolate(clahe_img.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze()

            # Store the result in the output tensor
            output[n, c] = clahe_img

    # Reshape output to the original shape
    output = output.view(*original_shape)

    return output