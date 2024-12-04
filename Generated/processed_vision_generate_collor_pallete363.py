def generate_rgb_colors(num_objects):
    palette = [33554431, 32767, 2097151]
    colors = []

    for i in range(num_objects):
        r = (i * palette[0]) % 255
        g = (i * palette[1]) % 255
        b = (i * palette[2]) % 255
        colors.append((r, g, b))

    return colors

