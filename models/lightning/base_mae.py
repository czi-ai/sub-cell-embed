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
class MAEPoolOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    pool_op: torch.FloatTensor = None
    pool_op2: torch.FloatTensor = None
    pool_attn: torch.FloatTensor = None
    pool_attn2: torch.FloatTensor = None
    pred_output: Optional[Tuple[torch.FloatTensor]] = None
    pred_output2: Optional[Tuple[torch.FloatTensor]] = None


class BaseMAE(L.LightningModule):
    def __init__(
        self,
        save_folder: str,
        encoder: nn.Module,
        decoder: nn.Module,
        pool_model: nn.Module = None,
        decoder_only_prot: bool = False,
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
        self.transforms = transforms if transforms is not None else None
        self.transforms2 = transforms2 if transforms2 is not None else None
        self.valid_transforms = valid_transforms

        print(
            f"Transforms: {self.transforms} \nTransforms2: {self.transforms2} \nValid Transforms: {self.valid_transforms}"
        )

        ## model components
        self.encoder = encoder
        self.mask_ratio = encoder.config.mask_ratio
        self.patch_size = encoder.config.patch_size

        ## pool model
        self.pool_model = MeanPool() if pool_model is None else pool_model
        self.pool_model = self.pool_model

        self.decoder = decoder

        self.only_prot = decoder_only_prot

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
            encoder.config.hidden_size if pool_model is None else pool_model.out_dim
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

    def patchify(self, pixel_values):
        patch_size, num_channels = self.patch_size, self.encoder.config.num_channels

        # patchify
        batch_size = pixel_values.shape[0]
        num_patches_one_direction = pixel_values.shape[2] // patch_size
        patchified_pixel_values = pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_one_direction,
            patch_size,
            num_patches_one_direction,
            patch_size,
        )
        patchified_pixel_values = torch.einsum(
            "nchpwq->nhwpqc", patchified_pixel_values
        )
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_one_direction * num_patches_one_direction,
            patch_size**2 * num_channels,
        )
        return patchified_pixel_values

    def unpatchify(self, patchified_pixel_values):
        patch_size, num_channels = self.patch_size, self.encoder.config.num_channels
        num_patches_one_direction = int(patchified_pixel_values.shape[1] ** 0.5)

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_one_direction,
            num_patches_one_direction,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum(
            "nhwpqc->nchpwq", patchified_pixel_values
        )
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_one_direction * patch_size,
            num_patches_one_direction * patch_size,
        )
        return pixel_values

    def restore_features(self, z, ids_restore, h, w):
        b = z.shape[0]
        masked_z = torch.zeros(
            (b, h * w - z.shape[1], z.shape[2]), device=z.device, dtype=z.dtype
        )
        z = torch.cat([z, masked_z], dim=1)
        z = (
            torch.gather(
                z, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, z.shape[2])
            )
            .reshape(b, h, w, -1)
            .permute(0, 3, 1, 2)
        )
        return z

    def configure_optimizers(self) -> optim.Optimizer:
        params = param_groups_weight_decay(
            self.encoder, self.weight_decay
        ) + param_groups_weight_decay(self.decoder, self.weight_decay)
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
                x2 = self.transforms2(x2)
            return x1, x2, protein_id, y, mask
        else:
            x1, x2, protein_id, y, mask = batch
            if self.valid_transforms is not None:
                x1 = self.valid_transforms(x1)
            return x1, x2, protein_id, y, mask

    def recon_loss(
        self,
        pixel_values: torch.Tensor,
        pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = self.patchify(pixel_values)

        if self.encoder.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / ((var + 1.0e-6) ** 0.5)

        if self.only_prot:
            pred = self.unpatchify(pred)
            target = self.unpatchify(target)
            loss = (pred[:, -1] - target[:, -1]) ** 2
            if mask is not None:
                mask = self.unpatchify(
                    mask.unsqueeze(-1).repeat(
                        1, 1, (self.patch_size**2) * pred.shape[1]
                    )
                )
            else:
                mask = torch.ones_like(loss)
        else:
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
            if mask is None:
                mask = torch.ones_like(loss)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss

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

        pred = self.online_finetuner(pool_op.detach())

        attentions = outputs.attentions
        if isinstance(attentions, tuple):
            if attentions[0] is None:
                attentions = None

        return MAEPoolOutput(
            last_hidden_state=outputs.last_hidden_state,
            mask=outputs.mask,
            ids_restore=outputs.ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=attentions,
            pool_op=pool_op,
            pool_attn=pool_attn,
            pred_output=pred,
        )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x1, _, _, y, mask = batch

        enc_outputs1 = self.forward(x1, object_mask=mask)

        ## reconstruction loss
        x1_pred = self.decoder(enc_outputs1.last_hidden_state, enc_outputs1.ids_restore)
        reconloss = self.recon_loss(x1, x1_pred.logits, enc_outputs1.mask)
        self.log("train_loss/reconloss", reconloss, on_step=True, on_epoch=False)

        clsloss = self.cls_criteria(enc_outputs1.pred_output, y)
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

        loss = reconloss + clsloss

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, _, _, y, mask = batch

        enc_outputs, clsloss, recon_loss, all_preds = self.validation_forward_minibatch(
            x, y, return_feat=True if batch_idx % 5 == 0 else False
        )
        self.log(
            "val_loss/reconloss", recon_loss, on_step=False, on_epoch=True, logger=True
        )
        self.log(
            "val_loss/online_val_loss",
            clsloss,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        if self.global_rank == 0 and batch_idx % 5 == 0:
            _, _, h, w = x.shape
            h_feat = h // self.patch_size
            w_feat = w // self.patch_size
            feat = self.restore_features(
                enc_outputs.last_hidden_state[:, 1:, :],
                enc_outputs.ids_restore,
                h_feat,
                w_feat,
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
            x_pred = self.unpatchify(all_preds)
            if self.decoder.config.norm_pix_loss:
                x_pred = torch.clamp(
                    (x_pred * x.std(dim=1, keepdim=True)) + x.mean(dim=1, keepdim=True),
                    0,
                    1,
                )
            if enc_outputs.attentions is not None:
                attn = enc_outputs.attentions[-1][:, :, 0, 1:]
                attn = self.restore_features(
                    attn.permute(0, 2, 1), enc_outputs.ids_restore, h_feat, w_feat
                )
                attn = F.interpolate(
                    attn, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
                save_overlay_attn(
                    x,
                    x_pred,
                    attn,
                    self.current_epoch,
                    batch_idx,
                    self.color_channels,
                    self.save_folder,
                    tag="ipattn",
                )
            if enc_outputs.pool_attn is not None:
                attn = self.restore_features(
                    enc_outputs.pool_attn[:, :, 1:].permute(0, 2, 1),
                    enc_outputs.ids_restore,
                    h_feat,
                    w_feat,
                )
                attn = F.interpolate(
                    attn, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
                save_overlay_attn(
                    x,
                    x_pred,
                    attn,
                    self.current_epoch,
                    batch_idx,
                    self.color_channels,
                    self.save_folder,
                    tag="pool_attn",
                )
            if enc_outputs.pool_attn2 is not None:
                attn = self.restore_features(
                    enc_outputs.pool_attn2[:, :, 1:].permute(0, 2, 1),
                    enc_outputs.ids_restore,
                    h_feat,
                    w_feat,
                )
                attn = F.interpolate(
                    attn, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
                save_overlay_attn(
                    x,
                    x_pred,
                    attn,
                    self.current_epoch,
                    batch_idx,
                    self.color_channels,
                    self.save_folder,
                    tag="pool_attn2",
                )

        return {
            "loss": recon_loss + clsloss,
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
        all_preds = []
        all_recon_loss = 0.0
        all_cls_loss = 0.0
        for i in range(len(x_split)):
            enc_outputs = self.forward(x_split[i], mask_ratio=0.0)
            all_enc_outputs.append(enc_outputs)

            out = self.decoder(enc_outputs.last_hidden_state, enc_outputs.ids_restore)
            x_pred = out.logits
            all_preds.append(x_pred)

            recon_loss = self.recon_loss(x_split[i], x_pred)
            all_recon_loss += recon_loss

            clsloss = self.cls_criteria(enc_outputs.pred_output, y_split[i])
            all_cls_loss += clsloss

        all_enc_outputs = MAEPoolOutput(
            last_hidden_state=torch.cat(
                [x.last_hidden_state for x in all_enc_outputs], dim=0
            ),
            ids_restore=torch.cat([x.ids_restore for x in all_enc_outputs], dim=0),
            mask=torch.cat([x.mask for x in all_enc_outputs], dim=0),
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
            pool_attn2=(
                None
                if all_enc_outputs[0].pool_attn2 is None
                else torch.cat([x.pool_attn2 for x in all_enc_outputs], dim=0)
            ),
        )

        all_preds = torch.cat(all_preds, dim=0)

        all_recon_loss /= len(x_split)
        all_cls_loss /= len(x_split)

        return all_enc_outputs, all_cls_loss, all_recon_loss, all_preds
