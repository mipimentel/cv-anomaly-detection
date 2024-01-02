import math
from typing import Callable, List, Union

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA, IncrementalPCA


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


def plot_multiscale_basic_features(
    image: Union[npt.NDArray[np.uint8], Image.Image],
    features_func: Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.float64]],
) -> None:
    """
    Generate a plot displaying multiscale basic features of an image.

    Parameters:
        image: Union[npt.NDArray[np.uint8], Image.Image]
            - The image to generate the features for. It can either be a numpy array of type uint8 or a PIL Image object.
        features_func: Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.float64]]
            - The function that calculates the features for the given image. It takes in a numpy array of type uint8 and returns a numpy array of type float64.

    Returns:
        None
    """

    if isinstance(image, Image.Image):
        image = np.array(image)
        # convert to default opencv BGR if it is a pillow image
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    image = np.array(image)
    # convert to default opencv BGR if it is a pillow image
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    features = features_func(image)
    num_features = features.shape[-1]

    # Calculate the grid size
    grid_size = int(np.ceil(np.sqrt(num_features)))

    # Create a figure with subplots in a square grid
    fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(15, 15))

    # Flatten the array of axes for easy iteration
    axes = axes.flatten()

    # Loop over all possible grid positions
    for i in range(grid_size**2):
        ax = axes[i]
        if i < num_features:
            # Select the ith feature across all scales
            feature_image = features[..., i]
            ax.imshow(feature_image, cmap="gray")
            ax.set_title(f"Feature {i + 1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_pca_cumulative_variance(
    pca: Union[PCA, IncrementalPCA], threshold: float = 0.9
) -> None:
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    threshold_component = np.where(cumulative_variance >= threshold)[0][0] + 1

    sns.barplot(
        x=np.arange(1, len(pca.explained_variance_ratio_) + 1), y=cumulative_variance
    )

    plt.axhline(y=threshold, color="red", linestyle="--")
    plt.text(
        x=threshold_component,
        y=threshold,
        s=f"  {threshold_component} components explain {threshold * 100}% variance",
        color="red",
    )

    plt.title("Explained Cumulative Variance Ratio by PCA Components")
    plt.xlabel("Principal Components")
    plt.ylabel("Cumulative Variance Ratio")
    plt.show()
: