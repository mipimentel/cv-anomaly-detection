# --------------------------------------------------------
# TinyViT Model Architecture
# Copyright (c) 2022 Microsoft
# Adapted from:
#   LeViT: https://github.com/microsoft/Cream/blob/main/TinyViT/inference.py
# Build the TinyViT Model
# --------------------------------------------------------


"""Model Inference."""
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from timm.data import create_transform
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from torchvision import datasets, transforms

from cv_anomaly_detection.img_dataframe import ImageDataFrameMVTEC

# from data import build_transform, imagenet_classnames
from cv_anomaly_detection.models.config import get_config
from cv_anomaly_detection.models.tiny_vit import tiny_vit_21m_224
from cv_anomaly_detection.utils import DATA_DIR, MVTEC_AD
from cv_anomaly_detection.utils.plots import (
    plot_features,
    plot_multiscale_basic_features,
    plot_pca_cumulative_variance,
)

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == "bicubic":
            return InterpolationMode.BICUBIC
        elif method == "lanczos":
            return InterpolationMode.LANCZOS
        elif method == "hamming":
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR

except:
    from timm.data.transforms import _pil_interp

config = get_config()


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32

    # RGB: mean, std
    rgbs = dict(
        default=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        inception=(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD),
        clip=(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    )
    mean, std = rgbs[config.DATA.MEAN_AND_STD_TYPE]

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        create_transform_t = (
            create_transform if not config.DISTILL.ENABLED else create_transform_record
        )
        transform = create_transform_t(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=(
                config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None
            ),
            auto_augment=(
                config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != "none" else None
            ),
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                config.DATA.IMG_SIZE, padding=4
            )

        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(
                    size, interpolation=_pil_interp(config.DATA.INTERPOLATION)
                ),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize(
                    (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                    interpolation=_pil_interp(config.DATA.INTERPOLATION),
                )
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(t)
    return transform


hdf5_db_path = os.path.join(DATA_DIR, "mnt_tiny_vit_tmp.h5")


if __name__ == "__main__":
    # Build model without classification head
    model = tiny_vit_21m_224(pretrained=True, num_classes=0)

    model.eval()

    img_df = ImageDataFrameMVTEC(category="cable")
    img_df.init_default()

    test_df = img_df.df[img_df.df["dataset_type"] == "test"]
    train_df = img_df.df[img_df.df["dataset_type"] == "train"]

    # Load Image
    img_path = train_df["image_path"].iloc[0]
    print(img_path)
    image = Image.open(img_path)
    transform = build_transform(is_train=False, config=config)

    # (1, 3, img_size, img_size)
    batch = transform(image)[None]

    with torch.no_grad():
        outputs = model(batch)

    print(outputs.shape)
    # print(outputs["norm_head_output"].shape)
    rows, _ = train_df.shape

    with h5py.File(hdf5_db_path, "w") as h5:
        h5.create_dataset(
            "data",
            shape=(rows, outputs.flatten().shape[0]),
            dtype=outputs.numpy().dtype,
        )

        for i in range(rows):
            img_path = train_df.iloc[i]["image_path"]
            image = Image.open(img_path)
            batch = transform(image)[None]
            with torch.no_grad():
                features = model(batch)

            flattened_feats = features.flatten()
            h5["data"][i, :] = flattened_feats

    min_val = np.inf
    max_val = -np.inf
    with h5py.File(hdf5_db_path, "r") as h5:
        data = h5["data"]
        print(data.shape)
        pca = PCA()
        min_val = min(np.min(data), min_val)  # Update the minimum value
        max_val = max(np.max(data), max_val)

        transformed_data = pca.fit_transform(data)

    print(f"Minimum value:{min_val}, Maximum value: {max_val}")
    print(pca.components_.shape)

    plot_pca_cumulative_variance(pca, threshold=0.99, remove_x_ticks=True)

    print(
        f"Mean from PCA reconstruction diff from original and reconstructed at 0 index: {np.mean(pca.inverse_transform(transformed_data[0]) - outputs.numpy().reshape(-1))}"
    )
    # for sanity check purpose of residual conversion difference
    print(
        f"Mean from PCA reconstruction diff from reconstructed at index 0  and reconstructed at index 1: {np.mean(pca.inverse_transform(transformed_data[1]) - outputs.numpy().reshape(-1))}"
    )

    # NPCA calculation

    threshold = 0.99
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    component_idx = np.argmax(cumulative_variance >= threshold)
    print(f"Component index threshold: {component_idx}")
    pca_components = pca.components_[:component_idx]

    with h5py.File(hdf5_db_path, "r") as h5:
        data = h5["data"]
        pca_transformed_data = np.dot(data - pca.mean_, pca_components.T)
        print(pca_transformed_data.shape, pca_components.shape)
        reconstructed = np.dot(pca_transformed_data, pca_components) + pca.mean_

    npca_components = pca.components_[component_idx:]

    with h5py.File(hdf5_db_path, "r") as h5:
        data = h5["data"]
        npca_transformed_data = np.dot(data - pca.mean_, npca_components.T)
        print(npca_transformed_data.shape, npca_components.shape)
        reconstructed = np.dot(npca_transformed_data, npca_components) + pca.mean_

    # Ledoit Wolf
    cov_npca_1_percent = LedoitWolf().fit(npca_transformed_data).covariance_
    cov_pca_99_percent = LedoitWolf().fit(pca_transformed_data).covariance_

    mean_npca_1_percent = np.mean(npca_transformed_data, axis=0)
    mean_pca_99_percent = np.mean(pca_transformed_data, axis=0)

    inv_cov_npca_1_percent = np.linalg.inv(cov_npca_1_percent)
    inv_cov_pca_99_percent = np.linalg.inv(cov_pca_99_percent)

    for idx, row in test_df.iterrows():
        path = row["image_path"]
        print(path)
        image = Image.open(path)
        batch = transform(image)[None]
        with torch.no_grad():
            features = model(batch)

        feats = features.flatten()
        # NPCA 1%
        npca_transformed_data = np.dot(feats - pca.mean_, npca_components.T)
        test_df.loc[idx, "mahalanobis_npca_1%"] = mahalanobis(
            npca_transformed_data, mean_npca_1_percent, inv_cov_npca_1_percent
        )

        # PCA 99%
        pca_transformed_data = np.dot(feats - pca.mean_, pca_components.T)
        test_df.loc[idx, "mahalanobis_pca_99%"] = mahalanobis(
            pca_transformed_data, mean_pca_99_percent, inv_cov_pca_99_percent
        )

    # visualize mahalanobis distance metrics
    print(test_df[["mahalanobis_pca_99%", "mahalanobis_npca_1%"]].describe())
    group_class = test_df.groupby("class")

    maha_pca_99 = group_class["mahalanobis_pca_99%"].describe()
    maha_npca_1 = group_class["mahalanobis_npca_1%"].describe()

    pd.set_option("display.max_columns", None)  # Show all columns
    print("\nDescriptive Statistics for Mahalanobis distance with PCA 99%")
    print(maha_pca_99)

    print("\nDescriptive Statistics for Mahalanobis distance with NPCA 1%")
    print(maha_npca_1)

    # plots for mahalanobis distance PCA 99%
    sns.violinplot(
        x="class",
        y="mahalanobis_pca_99%",
        data=test_df,
        hue="class",
        palette="colorblind",
    )
    sns.stripplot(
        x="class", y="mahalanobis_pca_99%", data=test_df, color="k", alpha=0.7
    )

    plt.xticks(rotation=90)
    plt.show()

    # plots for mahalanobis distance NPCA 1%
    sns.violinplot(
        x="class",
        y="mahalanobis_npca_1%",
        data=test_df,
        hue="class",
        palette="colorblind",
    )
    sns.stripplot(
        x="class", y="mahalanobis_npca_1%", data=test_df, color="k", alpha=0.7
    )

    plt.xticks(rotation=90)
    plt.show()
