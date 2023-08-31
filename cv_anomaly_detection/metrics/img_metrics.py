import cv2 as cv
import numpy as np
import numpy.typing as npt


def measure_brightness(image: npt.NDArray[np.uint8]) -> float:
    """
    Calculate the brightness of an image.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        float: The brightness value of the image.
    """
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    brightness = cv.mean(v)[0]

    return brightness
