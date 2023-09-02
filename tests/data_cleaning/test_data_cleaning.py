import os

import numpy as np
from PIL import Image

from cv_anomaly_detection.data_cleaning import HashDB


def test_hash_db():
    """Test HashDB."""

    central_square_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    central_square_image = central_square_image.astype(np.uint8)
    # white square in center
    central_square_image[25:50, 25:50] = 255
    duplicate = central_square_image.copy()

    cs_img_upper = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cs_img_upper[15:40, 25:50] = 255

    # convert to PIL
    central_square_image = Image.fromarray(central_square_image)
    duplicate = Image.fromarray(duplicate)

    db = HashDB("db.shelve", "dhash")

    assert os.path.exists("db.shelve")
    db[duplicate] = "File B"
    db[central_square_image] = "File A"

    # todo: refactor to add image and search hash for better abstraction
    hash_csi = str(db.hash_algorithm(central_square_image))
    hash_duplicate = str(db.hash_algorithm(duplicate))

    assert hash_csi == hash_duplicate
    assert db[hash_csi] == "File A"
    assert db[hash_duplicate] == "File A"

    os.remove("db.shelve")
    # clean up
    assert not os.path.exists("db.shelve")
