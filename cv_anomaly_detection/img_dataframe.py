import glob
import os
import shelve

import imagehash
import pandas as pd
from PIL import Image

from cv_anomaly_detection.metrics import img_metrics
from cv_anomaly_detection.metrics.img_metrics import measure_brightness
from cv_anomaly_detection.utils import MVTEC_AD


class ImageDataFrameMVTEC:
    def __init__(self, category: str):
        datasets = [
            dir_ds
            for dir_ds in os.listdir(MVTEC_AD)
            if os.path.isdir(os.path.join(MVTEC_AD, dir_ds))
        ]

        self.category_db_path = os.path.join(MVTEC_AD, category)
        self.train = os.path.join(self.category_db_path, "train", "good")
        self.test = os.path.join(self.category_db_path, "test")

        if os.path.exists(os.path.join(self.category_db_path, "image_properties.csv")):
            self.df = pd.read_csv(
                os.path.join(self.category_db_path, "image_properties.csv")
            )
        else:
            # Initialize an empty DataFrame with desired columns
            columns = ["image_path", "width", "height", "dataset_type", "class"]
            self.df = pd.DataFrame(columns=columns)

    def init_default(self):
        # Initialize an empty DataFrame with desired columns
        columns = ["image_path", "width", "height", "dataset_type", "class"]
        self.df = pd.DataFrame(columns=columns)

        # Populate the DataFrame
        for image_name in os.listdir(self.train):
            image_path = os.path.join(self.train, image_name)

            # Skip non-image files
            if not image_path.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            # Open the image using PIL
            with Image.open(image_path) as img:
                width, height = img.size

            # Concat a new row to the DataFrame
            self.df = pd.concat(
                [
                    self.df,
                    pd.DataFrame(
                        [
                            {
                                "image_path": image_path,
                                "width": width,
                                "height": height,
                                "dataset_type": "train",
                                "class": "good",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

        test_classes = os.listdir(self.test)
        print(test_classes)
        for test_class in test_classes:
            class_dir = os.path.join(self.test, test_class)
            for img_path in os.listdir(class_dir):
                image_path = os.path.join(class_dir, img_path)
                # Skip non-image files
                if not image_path.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                with Image.open(image_path) as img_s:
                    width, height = img_s.size
                    # Concat a new row to the DataFrame
                self.df = pd.concat(
                    [
                        self.df,
                        pd.DataFrame(
                            [
                                {
                                    "image_path": image_path,
                                    "width": width,
                                    "height": height,
                                    "dataset_type": "test",
                                    "class": test_class,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

    def apply_metrics(self):
        self.df["dhash"] = self.df["image_path"].apply(
            lambda path: imagehash.dhash(Image.open(path))
        )
        self.df["brightness"] = self.df["image_path"].apply(
            lambda path: measure_brightness(Image.open(path))
        )

    def save_to_csv(self):
        self.df.to_csv(
            os.path.join(self.category_db_path, "image_properties.csv"), index=False
        )
