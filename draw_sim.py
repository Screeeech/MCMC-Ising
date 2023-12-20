from PIL import Image
import numpy as np
import imageio


def draw_spin_grid(x):
    runs, n = x.shape
    img = Image.new(mode="RGB", size=(n, runs))

    for i in range(runs):
        for j in range(n):
            if x[i, j] == 1:
                img.putpixel((j, i), (0, 0, 0))
            else:
                img.putpixel((j, i), (255, 255, 255))

    return img


def create_gif(xruns, filename, duration=0.1):
    images = []
    for i in range(xruns.shape[0]):
        images.append(draw_spin_grid(xruns[i, :, :]))
    imageio.mimsave(filename, images, duration=duration)
