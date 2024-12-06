import importlib

import einops
import torch
from torch import nn
import torch.nn.functional as F


class CentroidDiffusionEMALoss(nn.Module):
    def __init__(self, projector: dict, labd_brownian: float = 0.1, n_views: int = 2):
        super(CentroidDiffusionEMALoss, self).__init__()
        module = importlib.import_module("models")
        self.projector: nn.Module = getattr(module, projector["name"])(
            **projector["args"]
        )
        self.out_dim = self.projector.out_channels
        self.labd_brownian = labd_brownian
        self.normalize = F.normalize
        self.cosine_similarity = F.cosine_similarity
        self.n_views = n_views

    def get_centroid_loss(self, z_i, z_j):
        c_j = self.normalize(torch.mean(z_j, dim=0, keepdim=True), dim=1)

        dij = 2 - 2 * (z_i * c_j).sum(dim=1)
        loss = dij.mean()
        return loss

    def get_brownian_loss(self, z):
        n, d = z.shape
        gauss_noise = torch.randn(1, d, device=z.device)
        gauss_noise_norm = self.normalize(gauss_noise, dim=1)

        sim = self.cosine_similarity(z, gauss_noise_norm, dim=1)
        loss = sim.mean()
        return loss

    def forward(self, x_i, x_j, labels=None):
        z_i = self.normalize(self.projector(x_i), dim=1)

        if labels is not None:
            centroid_loss = torch.tensor(0.0, device=z_i.device)
            brownian_loss = torch.tensor(0.0, device=z_i.device)
            uniques = torch.unique(labels)
            for i, u in enumerate(uniques):
                u_idx = labels == u
                z_i_u = z_i[u_idx]
                x_j_u = x_j[u_idx]
                centroid_loss_u = self.get_centroid_loss(z_i_u, x_j_u)
                centroid_loss += centroid_loss_u
                if self.labd_brownian > 0:
                    brownian_loss_u = self.get_brownian_loss(z_i_u)
                    brownian_loss += brownian_loss_u
            centroid_loss /= len(uniques)
            brownian_loss /= len(uniques)
        else:
            z_i = einops.rearrange(z_i, "(b n) d -> b n d", n=self.n_views)
            x_j = einops.rearrange(x_j, "(b n) d -> b n d", n=self.n_views)
            centroid_loss = self.get_centroid_loss(z_i, x_j)
            if self.labd_brownian > 0:
                brownian_loss = self.get_brownian_loss(z_i)
            else:
                brownian_loss = torch.tensor(0.0, device=z_i.device)

        loss = centroid_loss + (self.labd_brownian * brownian_loss)
        return loss


class CentroidDiffusionLoss(nn.Module):
    def __init__(self, projector: dict, labd_brownian: float = 0.1, n_views: int = 2):
        super(CentroidDiffusionLoss, self).__init__()
        module = importlib.import_module("models")
        self.projector: nn.Module = getattr(module, projector["name"])(
            **projector["args"]
        )
        self.out_dim = self.projector.out_channels
        self.labd_brownian = labd_brownian
        self.normalize = F.normalize
        self.mse_loss = F.mse_loss
        self.cosine_similarity = F.cosine_similarity
        self.n_views = n_views

    def get_centroid_loss(self, z_i, z_j):
        c_i = torch.mean(z_i, dim=1, keepdim=True)
        c_j = torch.mean(z_j, dim=1, keepdim=True)

        d = 1 - self.cosine_similarity(c_i, c_j, dim=2)
        loss = d.mean()
        return loss

    def get_brownian_loss(self, z_i, z_j):
        z = torch.cat([z_i, z_j], dim=1)

        b, n, d = z.shape
        gauss_noise = torch.randn(b, 1, d, device=z_i.device)
        gauss_noise_norm = self.normalize(gauss_noise, dim=2)

        sim = 1 + self.cosine_similarity(z, gauss_noise_norm, dim=2)
        loss = sim.mean()
        return loss

    def forward(self, x_i, x_j, labels=None):
        z_i = self.normalize(self.projector(x_i), dim=1)
        z_j = self.normalize(self.projector(x_j), dim=1)

        if labels is not None:
            z_i = einops.rearrange(z_i, "(b n) d -> b n d", n=self.n_views)
            z_j = einops.rearrange(z_j, "(b n) d -> b n d", n=self.n_views)
        else:
            z_i = z_i.unsqueeze(1)
            z_j = z_j.unsqueeze(1)

        centroid_loss = self.get_centroid_loss(z_i, z_j)
        brownian_loss = self.get_brownian_loss(z_i, z_j)

        loss = centroid_loss + (self.labd_brownian * brownian_loss)
        return loss
