from typing import Union

import cv2 as cv
import numpy as np
import numpy.typing as npt
from PIL import Image


def measure_brightness(image: Union[npt.NDArray[np.uint8], Image.Image]) -> float:
    """
    Calculate the brightness of an image.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        float: The brightness value of the image.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
        # convert to default opencv BGR if it is a pillow image
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    brightness = cv.mean(v)[0]

    return brightness
