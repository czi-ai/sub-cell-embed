from typing import Callable, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from torch import nn
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2 import functional as F


class RemoveChannel(nn.Module):
    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[0]
        if torch.rand(1) < self.p and c > 2:
            channel_to_blacken = torch.randint(0, c - 1, (1,))
            x[channel_to_blacken] = 0
        return x


class RescaleProtein(nn.Module):
    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p and x.max() > 0:
            random_factor = (np.random.rand() * 2) / (x.max() + 1e-6)
            x[-1] = x[-1] * random_factor
        return x


class PerChannelColorJitter(nn.Module):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=None, hue=None, p=0.5):
        super().__init__()
        self.transform = v2.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                x[i] = self.transform(x[i][None, ...])
        return x


class PerChannelGaussianBlur(nn.Module):
    def __init__(self, kernel_size=7, sigma=(0.1, 2.0), p=0.5):
        super().__init__()
        self.transform = v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                x[i] = self.transform(x[i][None, ...])
        return x


class PerChannelAdjustSharpness(nn.Module):
    def __init__(self, sharpness_factor=2, p=0.5):
        super().__init__()
        self.transform = v2.RandomAdjustSharpness(sharpness_factor=sharpness_factor)
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                x[i] = self.transform(x[i][None, ...])
        return x


class GaussianNoise(nn.Module):
    def __init__(self, sigma_range: Tuple[float, float], p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.sigma_range = sigma_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        c = x.shape[0]
        if torch.rand(1) < self.p:
            sigma = (
                torch.rand((c, 1, 1), device=device)
                * (self.sigma_range[1] - self.sigma_range[0])
                + self.sigma_range[0]
            )
            return x + (torch.randn_like(x) * sigma)
        else:
            return x


class PerChannelRandomErasing(nn.Module):
    def __init__(self, scale=(0.02, 0.1), ratio=(0.3, 3.3), p=0.5):
        super().__init__()
        self.transform = v2.RandomErasing(scale=scale, ratio=ratio)
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                x[i] = self.transform(x[i][None, ...])
        return x


class PerBatchCompose(Transform):
    def __init__(self, transforms: Sequence[Callable]) -> None:
        super().__init__()
        self.transforms = transforms

    def get_masked_transforms(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_t = [[] for _ in range(x.shape[0])]
        mask_t = [[] for _ in range(x.shape[0])]
        for i in range(x.shape[0]):
            x_i = x[i]
            mask_i = mask[i]
            for transform in self.transforms:
                x_i, mask_i = transform(x_i, mask_i)
            x_t[i] = x_i
            mask_t[i] = mask_i
        x_t = torch.stack(x_t)
        mask_t = torch.stack(mask_t)
        return x_t, mask_t

    def get_transforms(self, x):
        x_t = [[] for _ in range(x.shape[0])]
        for i in range(x.shape[0]):
            x_i = x[i]
            for transform in self.transforms:
                x_i = transform(x_i)
            x_t[i] = x_i
        x_t = torch.stack(x_t)
        return x_t

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            return self.get_masked_transforms(x, mask)
        else:
            return self.get_transforms(x)

    def __repr__(self) -> str:
        format_string = []
        for t in self.transforms:
            format_string.append(f"    {t}")
        return self.__class__.__name__ + "(\n" + "\n".join(format_string) + "\n)"


class PerChannelCompose(Transform):
    def __init__(self, transforms: Sequence[Callable]) -> None:
        super().__init__()
        self.transforms = transforms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        reshape_x = x.view(b * c, 1, h, w)
        for i in range(b * c):
            for transform in self.transforms:
                reshape_x[i] = transform(reshape_x[i])
        trans_x = reshape_x.view(b, c, h, w)
        return trans_x

    def __repr__(self) -> str:
        format_string = []
        for t in self.transforms:
            format_string.append(f"    {t}")
        return self.__class__.__name__ + "(\n" + "\n".join(format_string) + "\n)"
