import numpy as np
import pytest
from PIL import Image

from cv_anomaly_detection.metrics.img_metrics import measure_brightness


def test_measure_brightness():
    """Test measure_brightness."""

    bright = np.ones((100, 100, 3), dtype=np.uint8) * 255
    bright_brightness = measure_brightness(bright.astype(np.uint8))

    assert bright_brightness == 255

    dark = np.zeros((100, 100, 3), dtype=np.uint8)
    dark_brightness = measure_brightness(dark)

    assert dark_brightness == 0

    img_rnd = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    img_rnd_pil = Image.fromarray(img_rnd)

    rnd_brightness = measure_brightness(img_rnd)
    rnd_brightness_pil = measure_brightness(img_rnd_pil)

    assert isinstance(rnd_brightness, float)
    assert 255 > rnd_brightness > 0
    assert rnd_brightness_pil == rnd_brightness
