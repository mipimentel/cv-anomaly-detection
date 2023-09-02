# based on realpython article:
# https://realpython.com/fingerprinting-images-for-near-duplicate-detection/
import glob
import os
import shelve
from typing import Any

import imagehash
import numpy.typing as npt
from numpy import uint8
from PIL import Image


class HashDB:
    """A hash-based database."""

    def __init__(self, db_path: str, hash_algorithm: str) -> None:
        self.db = shelve.open(db_path)
        self.hash_algorithm = getattr(imagehash, hash_algorithm)
        # imagehash.dhash()

    def __getitem__(self, key: str) -> Any:
        """
        Get the value associated with the given key from the database.

        Parameters:
            key (str): The key to retrieve the value for.
        Returns:
            Any: The value associated with the given key.
        """
        return self.db[key]

    def __setitem__(self, key: Image, value) -> None:
        """
        Set the value of an item in the database.

        Parameters:
            key (numpy.ndarray): The key will be the hash of a given image.
            value: The value to be set for the item.

        Returns:
            None
        """
        self.db[str(self.hash_algorithm(key))] = value

    def __close__(self) -> None:
        """
        Synchronize and close the persistent database.
        Returns:
            None.
        """
        self.db.close()
