import math
from typing import List

import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image


def show(image: Image.Image) -> None:
    """
    Display an image.

    Args:
        image (PIL.Image.Image): The image to be displayed.

    Returns:
        None
    """
    # Figure size in inches
    plt.figure(figsize=(15, 5))

    plt.imshow(image, interpolation="bicubic")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def show_list_of_images(images: List[Image.Image]) -> None:
    num_images = len(images)
    grid_nrow = int(math.ceil(math.sqrt(num_images)))
    grid_ncol = int(math.ceil(num_images / grid_nrow))

    # Create subplots
    fig, axes = plt.subplots(grid_nrow, grid_ncol, figsize=(10, 10))

    # Flatten the axes for easier iteration
    axes = axes.flatten() if grid_nrow > 1 else [axes]

    for i in range(grid_nrow * grid_ncol):
        ax = axes[i]
        ax.axis("off")
        if i < num_images:
            ax.imshow(images[i], interpolation="bicubic")

    plt.tight_layout()
    plt.show()
