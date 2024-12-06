import importlib

import einops
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def get_contrastive_loss(features, temperature, labels=None):
    batch_size = features.shape[0]
    device = features.device

    if labels is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    else:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Num of labels does not match num of features")
        mask = torch.eq(labels, labels.T).float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(contrast_feature, contrast_feature.T), temperature
    )
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(contrast_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
        0,
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = -(mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = mean_log_prob_pos.mean()
    return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, projector: dict, temperature: float = 0.1):
        super(ContrastiveLoss, self).__init__()
        module = importlib.import_module("models")
        self.projector: nn.Module = getattr(module, projector["name"])(
            **projector["args"]
        )
        self.temperature = temperature
        self.normalize = F.normalize
        self.out_dim = self.projector.out_channels

    def forward(self, x_i, x_j, labels=None):
        z_i = self.normalize(self.projector(x_i), dim=1)
        z_j = self.normalize(self.projector(x_j), dim=1)

        feat = torch.stack([z_i, z_j], dim=1)
        loss = get_contrastive_loss(feat, self.temperature, labels)
        return loss
