import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics.functional as tmf
from timm.optim.optim_factory import param_groups_weight_decay
from torch import nn, optim
from torcheval.metrics.functional import (
    multilabel_accuracy,
    multilabel_auprc,
    topk_multilabel_accuracy,
)


from .base_mae import BaseMAE, MAEPoolOutput


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class ContrastBYOLMAEDecoupleAttn(BaseMAE):
    def __init__(
        self,
        supcon_model: nn.Module,
        ssl_model: nn.Module = None,
        momentum: float = 0.999,
        weight_recon: float = 1.0,
        weight_ssl: float = 0.05,
        weight_supcon: float = 0.0,
        mask_ratio2: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        ## model components
        self.pool_model2 = copy.deepcopy(kwargs["pool_model"])

        assert (ssl_model is not None) or (
            supcon_model is not None
        ), "At least one of the SSL or SupCon model should be provided"

        self.ssl_model = ssl_model
        self.supcon_model = supcon_model

        ## loss weights
        self.weight_recon = weight_recon
        self.weight_ssl = weight_ssl
        self.weight_supcon = weight_supcon

        self.momentum = momentum

        self.projection_head = nn.Sequential(
            nn.Linear(self.finetune_dim, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, supcon_model.out_dim),
        )
        self.setup_ema(self.encoder, self.pool_model, self.projection_head)

        self.online_finetuner2 = copy.deepcopy(self.online_finetuner)

        self.mask_ratio2 = mask_ratio2

    def setup_ema(self, encoder, pool_model, projection_head):
        self.target_encoder = copy.deepcopy(encoder)
        self.target_encoder.load_state_dict(encoder.state_dict())
        set_requires_grad(self.target_encoder, False)

        self.target_pool_model = copy.deepcopy(pool_model)
        self.target_pool_model.load_state_dict(pool_model.state_dict())
        set_requires_grad(self.target_pool_model, False)

        self.target_projection_head = copy.deepcopy(projection_head)
        self.target_projection_head.load_state_dict(projection_head.state_dict())
        set_requires_grad(self.target_projection_head, False)

    def configure_optimizers(self):
        params = param_groups_weight_decay(
            self.encoder, self.weight_decay
        ) + param_groups_weight_decay(self.decoder, self.weight_decay)
        params += [{"params": self.pool_model.parameters()}]
        params += [{"params": self.pool_model2.parameters()}]
        params += [{"params": self.online_finetuner.parameters()}]
        params += [{"params": self.online_finetuner2.parameters()}]

        params += (
            [{"params": self.ssl_model.parameters()}]
            if self.ssl_model is not None
            else []
        )
        params += [{"params": self.supcon_model.parameters()}]
        params += [{"params": self.projection_head.parameters()}]

        optimizer = optim.AdamW(params, lr=self.lr, betas=self.betas)
        return optimizer

    @torch.no_grad()
    def on_before_zero_grad(self, _):
        for params_q, params_k in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            params_k.data = params_k.data * self.momentum + params_q.data * (
                1.0 - self.momentum
            )
        for params_q, params_k in zip(
            self.pool_model.parameters(), self.target_pool_model.parameters()
        ):
            params_k.data = params_k.data * self.momentum + params_q.data * (
                1.0 - self.momentum
            )
        for params_q, params_k in zip(
            self.projection_head.parameters(), self.target_projection_head.parameters()
        ):
            params_k.data = params_k.data * self.momentum + params_q.data * (
                1.0 - self.momentum
            )

    def gather_tensors(self, x, sync_grads=True):
        return self.all_gather(x, sync_grads=sync_grads).reshape(-1, x.shape[-1])

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = None,
        object_mask: torch.Tensor = None,
    ) -> MAEPoolOutput:
        output_attentions = True if not self.training else False
        mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio

        outputs = self.encoder(
            x,
            output_attentions=output_attentions,
            mask_ratio=mask_ratio,
            object_mask=object_mask,
        )
        pool_op, pool_attn = self.pool_model(outputs.last_hidden_state)
        pool_op2, pool_attn2 = self.pool_model2(outputs.last_hidden_state)

        pred = self.online_finetuner(pool_op.detach())
        pred2 = self.online_finetuner2(pool_op2.detach())

        attentions = outputs.attentions
        if isinstance(attentions, tuple):
            if attentions[-1] is None:
                attentions = None

        return MAEPoolOutput(
            last_hidden_state=outputs.last_hidden_state,
            mask=outputs.mask,
            ids_restore=outputs.ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=attentions,
            pool_op=pool_op,
            pool_op2=pool_op2,
            pool_attn=pool_attn,
            pool_attn2=pool_attn2,
            pred_output=pred,
            pred_output2=pred2,
        )

    def training_step(self, batch, batch_idx):
        x1, x2, protein_id, y, mask = batch

        enc_outputs1 = self.forward(x1, object_mask=mask)
        enc_outputs2 = self.forward(x2, mask_ratio=self.mask_ratio2)

        ## contrastive loss
        if self.ssl_model is not None:
            feat1_gather = self.gather_tensors(enc_outputs1.pool_op2, sync_grads=True)
            feat2_gather = self.gather_tensors(enc_outputs2.pool_op2, sync_grads=True)

            sslconloss = self.ssl_model(feat1_gather, feat2_gather)
            self.log("train_loss/sslconloss", sslconloss, on_step=True, on_epoch=False)
        else:
            sslconloss = torch.tensor(0.0, device=self.device)

        ## byol loss
        online_feat1_gather = self.gather_tensors(
            self.projection_head(enc_outputs1.pool_op)
        )
        online_feat2_gather = self.gather_tensors(
            self.projection_head(enc_outputs2.pool_op)
        )
        with torch.no_grad():
            target_op1 = self.target_encoder(x1)
            target_op2 = self.target_encoder(x2, mask_ratio=self.mask_ratio2)

            target_feat1, _ = self.target_pool_model(target_op1.last_hidden_state)
            target_feat2, _ = self.target_pool_model(target_op2.last_hidden_state)

            target_feat1_gather = self.gather_tensors(
                self.target_projection_head(target_feat1)
            )
            target_feat2_gather = self.gather_tensors(
                self.target_projection_head(target_feat2)
            )

        labels = self.all_gather(protein_id).reshape(-1)
        byol_loss = (
            self.supcon_model(online_feat1_gather, target_feat2_gather, labels)
            + self.supcon_model(online_feat2_gather, target_feat1_gather, labels)
        ) / 2.0
        self.log("train_loss/supconloss", byol_loss, on_step=True, on_epoch=False)

        ## reconstruction loss
        x1_pred = self.decoder(enc_outputs1.last_hidden_state, enc_outputs1.ids_restore)
        reconloss = self.recon_loss(x1, x1_pred.logits, enc_outputs1.mask)
        self.log("train_loss/reconloss", reconloss, on_step=True, on_epoch=False)

        ## mae loss
        mae_loss = (
            (self.weight_ssl * sslconloss)
            + (self.weight_supcon * byol_loss)
            + (self.weight_recon * reconloss)
        )
        self.log("train_loss/mae_loss", mae_loss, on_step=True, on_epoch=False)

        clsloss = (
            self.cls_criteria(enc_outputs1.pred_output, y)
            + self.cls_criteria(enc_outputs2.pred_output, y)
        ) / 2.0
        self.log("train_loss/online_cls_loss", clsloss, on_step=True, on_epoch=False)

        clsloss2 = (
            self.cls_criteria(enc_outputs1.pred_output2, y)
            + self.cls_criteria(enc_outputs2.pred_output2, y)
        ) / 2.0
        self.log("train_loss/online_cls_loss2", clsloss2, on_step=True, on_epoch=False)

        output = F.sigmoid(enc_outputs1.pred_output).float()
        y = y.int()

        ml_auprc = multilabel_auprc(
            output, y, num_labels=self.num_classes, average="macro"
        )
        self.log("train_metrics/ml_auprc", ml_auprc, on_step=True, on_epoch=False)
        ml_auprc2 = multilabel_auprc(
            F.sigmoid(enc_outputs1.pred_output2).float(),
            y,
            num_labels=self.num_classes,
            average="macro",
        )
        self.log("train_metrics/ml_auprc2", ml_auprc2, on_step=True, on_epoch=False)

        ml_topk_acc = topk_multilabel_accuracy(output, y, criteria="hamming", k=5)
        self.log(
            "train_metrics/ml_topk_acc",
            ml_topk_acc,
            on_step=True,
            on_epoch=False,
        )
        ml_topk_acc2 = topk_multilabel_accuracy(
            F.sigmoid(enc_outputs1.pred_output2).float(), y, criteria="hamming", k=5
        )
        self.log(
            "train_metrics/ml_topk_acc2",
            ml_topk_acc2,
            on_step=True,
            on_epoch=False,
        )

        loss = mae_loss + clsloss + clsloss2
        return loss
