from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn.functional as F
from timm.optim.optim_factory import param_groups_weight_decay
from torch import nn, optim
from torcheval.metrics.functional import (
    multilabel_accuracy,
    multilabel_auprc,
    topk_multilabel_accuracy,
)
from transformers.utils import ModelOutput

from models.focal_loss import SigmoidFocalLoss

from .save_utils import save_feat_nmf, save_overlay_attn


class MeanPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1), None


@dataclass
class ViTPoolOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    pool_op: torch.FloatTensor = None
    pool_attn: torch.FloatTensor = None
    pred_output: Optional[Tuple[torch.FloatTensor]] = None


class BaseSupervised(L.LightningModule):
    def __init__(
        self,
        save_folder: str,
        vit_model: nn.Module,
        pool_model: nn.Module = None,
        num_classes: int = 36,
        class_weights: torch.Tensor = None,
        categories: List[str] = None,
        max_epochs: int = 1000,
        warmup_epochs: int = 10,
        batches_per_epoch: int = 1000,
        init_lr: float = 1e-3,
        weight_decay: float = 0.05,
        betas: Tuple[float, float] = (0.9, 0.95),
        color_channels: List[str] = ["red", "green", "blue"],
        transforms: Optional[Callable] = None,
        transforms2: Optional[Callable] = None,
        valid_transforms: Optional[Callable] = None,
    ):
        super().__init__()

        ## transforms
        self.transforms = transforms
        self.transforms2 = transforms2
        self.valid_transforms = valid_transforms

        ## model components
        self.encoder = vit_model
        self.patch_size = vit_model.config.patch_size

        ## pool model
        self.pool_model = MeanPool() if pool_model is None else pool_model

        ## save folders
        self.save_folder = f"{save_folder}/valid_images"

        ## training params
        self.max_epochs = max_epochs
        self.batches_per_epoch = batches_per_epoch
        self.lr = init_lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_epochs = warmup_epochs

        self.color_channels = color_channels

        ## classification head and loss
        self.finetune_dim = (
            vit_model.config.hidden_size if pool_model is None else pool_model.out_dim
        )
        self.online_finetuner = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.finetune_dim, self.finetune_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.finetune_dim // 2, num_classes),
        )
        # self.cls_criteria = nn.BCEWithLogitsLoss(weight=class_weights)
        self.cls_criteria = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
        self.categories = categories
        self.num_classes = num_classes

        self.save_hyperparameters()

    def configure_optimizers(self) -> optim.Optimizer:
        params = param_groups_weight_decay(self.encoder, self.weight_decay)
        params += [{"params": self.pool_model.parameters()}]
        params += [{"params": self.online_finetuner.parameters()}]

        optimizer = optim.AdamW(params, lr=self.lr, betas=self.betas)
        return optimizer

    def adjust_learning_rate_linear(self, step: int) -> float:
        max_steps = self.max_epochs * self.batches_per_epoch
        warmup_steps = self.warmup_epochs * self.batches_per_epoch
        base_lr = self.lr
        if step < warmup_steps:
            lr = base_lr * step / warmup_steps
        else:
            step -= warmup_steps
            max_steps -= warmup_steps
            q = 1 - step / max_steps
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1 - q)
        return lr

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: optim.Optimizer,
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        optimizer.step(closure=optimizer_closure)

        lr = self.adjust_learning_rate_linear(
            batch_idx + epoch * self.batches_per_epoch
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        self.log("lr", optimizer.param_groups[0]["lr"], on_step=True, on_epoch=False)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.training:
            x1, x2, protein_id, y, mask = batch
            if self.transforms is not None:
                if mask is not None:
                    x1, mask = self.transforms(x1, mask)
                else:
                    x1 = self.transforms(x1)
                x2 = self.transforms(x2)
            if self.transforms2 is not None:
                x1 = self.transforms2(x1)
                x2 = self.transforms2(x2)
            return x1, x2, protein_id, y, mask
        else:
            x1, x2, protein_id, y, mask = batch
            if self.valid_transforms is not None:
                x1 = self.valid_transforms(x1)
            return x1, x2, protein_id, y, mask

    def forward(self, x: torch.Tensor) -> ViTPoolOutput:
        output_attentions = True if not self.training else False

        outputs = self.encoder(x, output_attentions=output_attentions)
        pool_op, pool_attn = self.pool_model(outputs.last_hidden_state)

        pred = self.online_finetuner(pool_op)

        attentions = outputs.attentions
        if isinstance(attentions, tuple):
            if attentions[-1] is None:
                attentions = None

        return ViTPoolOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=attentions,
            pool_op=pool_op,
            pool_attn=pool_attn,
            pred_output=pred,
        )

    def gather_tensors(self, x, sync_grads=True):
        return self.all_gather(x, sync_grads=sync_grads).reshape(-1, x.shape[-1])

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x1, x2, protein_id, y, mask = batch

        enc_outputs1 = self.forward(x1)
        enc_outputs2 = self.forward(x2)

        clsloss = (
            self.cls_criteria(enc_outputs1.pred_output, y)
            + self.cls_criteria(enc_outputs2.pred_output, y)
        ) / 2.0
        self.log("train_loss/online_cls_loss", clsloss, on_step=True, on_epoch=False)

        output = F.sigmoid(enc_outputs1.pred_output).float()
        y = y.int()

        ml_auprc = multilabel_auprc(
            output, y, num_labels=self.num_classes, average="macro"
        )
        self.log("train_metrics/ml_auprc", ml_auprc, on_step=True, on_epoch=False)
        ml_topk_acc = topk_multilabel_accuracy(output, y, criteria="hamming", k=5)
        self.log(
            "train_metrics/ml_topk_acc",
            ml_topk_acc,
            on_step=True,
            on_epoch=False,
        )
        
        return clsloss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, _, _, y, mask = batch

        enc_outputs, clsloss = self.validation_forward_minibatch(
            x, y, return_feat=True if batch_idx % 5 == 0 else False
        )

        self.log(
            "val_loss/online_val_loss",
            clsloss,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        if self.global_rank == 0 and batch_idx % 5 == 0:
            b, _, h, w = x.shape
            h_feat = h // self.patch_size
            w_feat = w // self.patch_size
            feat = (
                enc_outputs.last_hidden_state[:, 1:, :]
                .permute(0, 2, 1)
                .reshape(b, -1, h_feat, w_feat)
            )
            save_feat_nmf(
                x,
                feat,
                self.current_epoch,
                batch_idx,
                self.color_channels,
                self.save_folder,
                tag="feat",
            )

            if enc_outputs.attentions is not None:
                attn = (
                    enc_outputs.attentions[-1][:, :, 0, 1:]
                    .permute(0, 2, 1)
                    .reshape(b, -1, h_feat, w_feat)
                )

                attn = F.interpolate(
                    attn, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
                save_overlay_attn(
                    x,
                    x,
                    attn,
                    self.current_epoch,
                    batch_idx,
                    self.color_channels,
                    self.save_folder,
                    tag="ipattn",
                )
            if enc_outputs.pool_attn is not None:
                attn = (
                    enc_outputs.pool_attn[:, :, 1:]
                    .permute(0, 2, 1)
                    .reshape(b, -1, h_feat, w_feat)
                )

                attn = F.interpolate(
                    attn, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
                save_overlay_attn(
                    x,
                    x,
                    attn,
                    self.current_epoch,
                    batch_idx,
                    self.color_channels,
                    self.save_folder,
                    tag="pool_attn",
                )

        return {
            "loss": clsloss,
            "output": enc_outputs.pred_output,
            "target": y,
            "features": enc_outputs.pool_op,
        }

    def validation_forward_minibatch(
        self, x: torch.Tensor, y: torch.Tensor, return_feat: bool = False
    ):
        x_split = torch.split(x, 32, dim=0)
        y_split = torch.split(y, 32, dim=0)

        all_enc_outputs = []
        all_cls_loss = 0.0
        for i in range(len(x_split)):
            enc_outputs = self.forward(x_split[i])
            all_enc_outputs.append(enc_outputs)

            clsloss = self.cls_criteria(enc_outputs.pred_output, y_split[i])
            all_cls_loss += clsloss

        all_enc_outputs = ViTPoolOutput(
            last_hidden_state=torch.cat(
                [x.last_hidden_state for x in all_enc_outputs], dim=0
            ),
            pred_output=torch.cat(
                [torch.sigmoid(x.pred_output) for x in all_enc_outputs]
            ),
            pool_op=torch.cat([x.pool_op for x in all_enc_outputs], dim=0),
            attentions=(
                None
                if all_enc_outputs[0].attentions is None
                else torch.cat([x.attentions for x in all_enc_outputs], dim=0)
            ),
            pool_attn=(
                None
                if all_enc_outputs[0].pool_attn is None
                else torch.cat([x.pool_attn for x in all_enc_outputs], dim=0)
            ),
        )

        all_cls_loss /= len(x_split)

        return all_enc_outputs, all_cls_loss
