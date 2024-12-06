import importlib
from typing import Any, Dict

import torchvision.transforms.v2 as v2
from torchvision.transforms.v2 import InterpolationMode

from utils.augmentations import (
    PerBatchCompose,
    PerChannelAdjustSharpness,
    PerChannelColorJitter,
    PerChannelGaussianBlur,
    PerChannelRandomErasing,
    RemoveChannel,
    RescaleProtein,
    GaussianNoise,
)

from streaming import Stream


def get_datasets(
    config: Dict[str, Any],
    train_device_microbatch_size: int,
    val_device_microbatch_size: int,
):

    transform_list = [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomChoice(
            [
                v2.RandomAffine(
                    degrees=90,
                    translate=(0.2, 0.2),
                    scale=(0.8, 1.2),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
                v2.RandomPerspective(
                    distortion_scale=0.25,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
            ]
        ),
    ]
    transform_list += (
        [
            v2.RandomResizedCrop(
                size=(448, 448),
                scale=(0.75, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=InterpolationMode.BILINEAR,
            ),
        ]
        if config["image_size"] != 448
        else []
    )

    transforms2_list = [
        RemoveChannel(p=0.25),
        RescaleProtein(p=0.25),
        PerChannelColorJitter(brightness=0.5, contrast=0.5, p=1.0),
        v2.RandomChoice(
            [
                PerChannelGaussianBlur(kernel_size=7, sigma=(0.1, 2.0), p=0.5),
                PerChannelAdjustSharpness(sharpness_factor=2, p=0.5),
            ]
        ),
        GaussianNoise(sigma_range=(0.01, 0.05), p=0.5),
        PerChannelRandomErasing(scale=(0.02, 0.1), ratio=(0.3, 3.3), p=0.5),
    ]

    dataset_config_dict = config["args"]

    transforms = PerBatchCompose(transform_list)
    transforms2 = (
        PerBatchCompose(transforms2_list)
        if dataset_config_dict["ssl_transform"]
        else None
    )
    valid_transforms = (
        PerBatchCompose([v2.CenterCrop(size=(448, 448))])
        if config["image_size"] != 448
        else None
    )

    dataset_module = importlib.import_module("data")

    if "stream" in config["dataset"].lower():
        train_streams = []
        for i in range(1, 5):
            stream = Stream(
                local=f".local/train{i}",
                remote=config["train_remote_path"].replace(
                    "hpa_mds", f"hpa_mds_{config['image_size']}"
                ),
                proportion=0.25,
            )
            train_streams.append(stream)
        train_dataset = getattr(dataset_module, config["dataset"])(
            streams=train_streams,
            shuffle=True,
            batch_size=train_device_microbatch_size,
            color_channels=dataset_config_dict["color_channels"],
            n_cells=dataset_config_dict["n_cells"],
            mask_prob=dataset_config_dict["mask_prob"],
            normalize=dataset_config_dict["normalize"],
            return_cell_mask=dataset_config_dict["return_cell_mask"],
        )
    else:
        train_dataset = getattr(dataset_module, config["dataset"])(
            local=".local/train",
            remote=config["train_remote_path"].replace(
                "hpa_mds", f"hpa_mds_{config['image_size']}"
            ),
            shuffle=True,
            batch_size=train_device_microbatch_size,
            color_channels=dataset_config_dict["color_channels"],
            n_cells=dataset_config_dict["n_cells"],
            mask_prob=dataset_config_dict["mask_prob"],
            normalize=dataset_config_dict["normalize"],
            return_cell_mask=dataset_config_dict["return_cell_mask"],
        )
    valid_dataset = getattr(dataset_module, config["dataset"].replace("Stream", ""))(
        local=".local/valid",
        remote=config["val_remote_path"].replace(
            "hpa_mds", f"hpa_mds_{config['image_size']}"
        ),
        shuffle=False,
        batch_size=val_device_microbatch_size,
        color_channels=dataset_config_dict["color_channels"],
        n_cells=-1,
        mask_prob=1.0,
        normalize=dataset_config_dict["normalize"],
    )

    return (train_dataset, valid_dataset), (transforms, transforms2, valid_transforms)
