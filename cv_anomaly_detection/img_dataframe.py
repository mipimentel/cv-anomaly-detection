import glob
import os
import shelve
from typing import Any, Callable, Hashable, Sequence, Tuple

import imagehash
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.series import Series
from PIL import Image

from cv_anomaly_detection.metrics import img_metrics
from cv_anomaly_detection.metrics.img_metrics import measure_brightness
from cv_anomaly_detection.utils import MVTEC_AD


class ImageDataFrameMVTEC:
    def __init__(self, category: str):
        """
        Initializes the class instance with the given category.

        Parameters:
            category (str): The category of the dataset.

        Returns:
            None
        """
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
        """
        Initializes a default DataFrame with desired columns and populates it with image data.

        Returns:
            None
        """

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

    def apply_metrics(self, metrics: Tuple[str, Callable[[str], Any]]):
        """
        Apply metrics to the dataframe.

        Parameters:
            metrics (Tuple[str, Callable[[str], Any]]): A tuple of column names and functions to apply to each value in the "image_path" column of the dataframe.

        Returns:
            None
        """
        for col, func in metrics:
            self.df[col] = self.df["image_path"].apply(func)

    def sample_images_plot(self, num_samples_per_label: int = 3):
        """
        Plots a grid of sample images for each class in the dataset.

        Parameters:
            num_samples_per_label (int): The number of sample images to plot for each label. Default is 3.

        Returns:
            None
        """
        # Number of unique labels
        num_labels = self.df["class"].nunique()
        labels = self.df["class"].unique()

        # Create subplots: one row for each label, three columns for the images
        fig, axes = plt.subplots(
            num_labels, num_samples_per_label, figsize=(15, 5 * num_labels)
        )

        # Iterate through the DataFrame and plot each set of 3 images per label
        for i, label in enumerate(labels):
            subset = self.df[self.df["class"] == label].sample(num_samples_per_label)
            for j, (_, row) in enumerate(subset.iterrows()):
                ax = axes[i, j] if num_labels > 1 else axes[j]
                ax.imshow(Image.open(row["image_path"]))
                ax.set_title(label)
                ax.axis("off")

        plt.tight_layout()
        plt.show()

    def find_duplicates(
        self, subset: Hashable | Sequence[Hashable] | None = None
    ) -> Series:
        """
        Find duplicates in the DataFrame.

        Parameters:
            subset (Hashable | Sequence[Hashable] | None, optional):
                A hashable value or sequence of hashable values to specify the columns
                to consider for detecting duplicates. If None, all columns are used.
                Defaults to None.

        Returns:
            Series:
                A boolean Series indicating whether each row is a duplicate or not.
        """
        return self.df[self.df.duplicated(subset=subset)]

    def save_to_csv(self):
        """
        Save the DataFrame to a CSV file.

        This function saves the DataFrame to a CSV file at the specified path.
        The CSV file will be named "image_properties.csv" and will be saved in the
        category database path.

        Returns:
            None
        """
        self.df.to_csv(
            os.path.join(self.category_db_path, "image_properties.csv"), index=False
        )
