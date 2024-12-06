from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn


class ProjectionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mlp_layers: List[int],
        add_bn: bool = True,
        avg_pool: bool = False,
        normalize: bool = False,
    ):
        super(ProjectionHead, self).__init__()
        self.in_channels = in_channels
        self.out_channels = mlp_layers[-1]
        self.mlp_layers = mlp_layers
        self.add_bn = add_bn
        self.avg_pool = avg_pool
        self.normalize = normalize
        self.mlp = self._init_mlp()

    def _init_mlp(self):
        layers = []
        if self.avg_pool:
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            layers.append(nn.Flatten(1))
        f = [self.in_channels] + self.mlp_layers
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            if self.add_bn:
                layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        if self.normalize:
            x = F.normalize(x, dim=1)
        return x


class ProjectionHeadCLIP(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)

        x += projected

        return self.layer_norm(x)
