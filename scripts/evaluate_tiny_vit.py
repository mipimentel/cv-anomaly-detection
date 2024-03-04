# --------------------------------------------------------
# TinyViT Model Architecture
# Copyright (c) 2022 Microsoft
# Adapted from:
#   LeViT: https://github.com/microsoft/Cream/blob/main/TinyViT/inference.py
# Build the TinyViT Model
# --------------------------------------------------------


"""Model Inference."""
import numpy as np
import torch
from PIL import Image
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


if __name__ == "__main__":
    outputs = {}  # A dictionary to store outputs

    def get_layer_output(module, input, output):
        outputs["norm_head_output"] = output

    # Attach the hook to your layer

    # Build model
    model = tiny_vit_21m_224(pretrained=True)
    hook = model.norm_head.register_forward_hook(get_layer_output)

    model.eval()

    img_df = ImageDataFrameMVTEC(category="cable")
    img_df.init_default()

    test_df = img_df.df[img_df.df["dataset_type"] == "test"]
    train_df = img_df.df[img_df.df["dataset_type"] == "train"]

    # Load Image
    img_path = train_df["image_path"].iloc[0]
    image = Image.open(img_path)
    transform = build_transform(is_train=False, config=config)

    # (1, 3, img_size, img_size)
    batch = transform(image)[None]

    with torch.no_grad():
        logits = model(batch)

    print(outputs["norm_head_output"].shape)
