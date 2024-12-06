import os
from typing import Any, Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from sklearn.metrics import (
    average_precision_score,
    label_ranking_average_precision_score,
)
from torcheval.metrics.functional import (
    multilabel_accuracy,
    multilabel_auprc,
    topk_multilabel_accuracy,
)

from models.lightning.save_utils import plot_feature_umaps
from utils.distributed import all_gather


class ResultSaveCallback(Callback):
    def __init__(self, plot_metrics: bool = False, plot_feats: bool = False):
        super().__init__()
        self.plot_metrics = plot_metrics
        self.plot_feats = plot_feats

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.metrics_folder = pl_module.save_folder.replace("valid_images", "metrics")
        self.map_path = f"{self.metrics_folder}/online_finetuning_map_df.csv"
        self.map_df = (
            pd.read_csv(self.map_path)
            if os.path.exists(self.map_path)
            else pd.DataFrame()
        )
        self.categories = pl_module.categories

        self.outputs = []
        self.targets = []

        if self.plot_feats:
            self.umap_folder = pl_module.save_folder.replace("valid_images", "umap")
            self.features = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[Union[torch.tensor, Mapping[str, Any]]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        self.outputs.append(outputs["output"])
        self.targets.append(outputs["target"])

        if self.plot_feats:
            self.features.append(outputs["features"])

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        outputs = torch.cat(self.outputs, dim=0)
        outputs = torch.cat(all_gather(outputs), dim=0).float()

        targets = torch.cat(self.targets, dim=0)
        targets = torch.cat(all_gather(targets), dim=0).int()

        ml_auprc = multilabel_auprc(
            outputs, targets, num_labels=len(self.categories), average="macro"
        )
        pl_module.log("val_metrics/total_ml_auprc", ml_auprc, logger=True)

        ml_topk_acc = topk_multilabel_accuracy(
            outputs, targets, criteria="hamming", k=5
        )
        pl_module.log("val_metrics/total_ml_topk_acc", ml_topk_acc, logger=True)

        if self.plot_metrics:
            self.save_metrics(outputs, targets, pl_module)

        if self.plot_feats:
            self.plot_features(targets, pl_module)

        self.outputs.clear()
        self.targets.clear()

    def save_metrics(
        self, outputs: torch.Tensor, targets: torch.Tensor, pl_module: LightningModule
    ):
        output1 = pd.DataFrame(outputs.cpu().numpy(), columns=self.categories)
        target1 = pd.DataFrame(targets.cpu().numpy(), columns=self.categories)

        mlrap = label_ranking_average_precision_score(target1, output1)
        pl_module.log("val_metrics/total_mlrap", mlrap, logger=True)

        mean_ap = average_precision_score(target1, output1, average="macro")
        pl_module.log("val_metrics/total_map", mean_ap, logger=True)

        avg_precisions = []
        for cat in self.categories:
            cat_avg_prec = average_precision_score(target1[cat], output1[cat])
            avg_precisions.append(cat_avg_prec)
        categories = self.categories + ["Overall"]
        mean_ap = average_precision_score(target1, output1, average="macro")
        avg_precisions.append(mean_ap)
        df_avg_prec = pd.DataFrame(
            np.array(avg_precisions)[None, ...], columns=categories
        )

        self.map_df = pd.concat([self.map_df, df_avg_prec], axis=0).reset_index(
            drop=True
        )
        self.map_df.to_csv(self.map_path, index=False)

        len_map_df = len(self.map_df)
        top20_cats = (
            self.map_df.iloc[max(0, len_map_df - 5) : len_map_df]
            .mean()
            .sort_values(ascending=False)
            .index[:21]
            .tolist()
        )
        top20_cats.append("Overall") if "Overall" not in top20_cats else None
        df_avg_prec = self.map_df[top20_cats]

        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(
            df_avg_prec, annot=True, ax=ax, cmap="vlag", fmt=".2f", vmin=0, vmax=1
        )
        ax.set_title(f"Online Finetuning MAP: {mean_ap:.3f}")
        plt.savefig(
            f"{self.metrics_folder}/online_finetuning_map.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_features(self, targets: torch.Tensor, pl_module: LightningModule):
        features = torch.cat(self.features, dim=0)
        features = torch.cat(all_gather(features), dim=0).float()

        features_dict = {
            "encoder": features.cpu().numpy(),
            "targets": targets.cpu().numpy(),
        }

        if pl_module.global_rank == 0:
            plot_feature_umaps(
                features_dict, ["encoder"], pl_module.current_epoch, self.umap_folder
            )
        self.features.clear()
