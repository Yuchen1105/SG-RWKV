# src/lightning/seg_module.py
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from monai.losses import DiceCELoss, DiceLoss
from omegaconf import DictConfig
from torchmetrics import MetricCollection
import pytorch_lightning as pl
import torch.nn.functional as F

from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, BinaryRecall, BinaryPrecision, \
    MulticlassF1Score, MulticlassJaccardIndex
from ..metrics.distance_metric import BinaryASD, BinaryHD95


class SegLitModule(pl.LightningModule):
    """Unified LightningModule for 2D medical image segmentation (binary & multiclass)."""

    def __init__(self, model: nn.Module, cfg: DictConfig) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters(ignore=["model"])

        # ---- task ----
        self.task = str(cfg.data.task.lower())  # "binary" | "multiclass"
        if self.task not in ("binary", "multiclass"):
            raise ValueError(f"Unknown cfg.data.task='{self.task}', must be 'binary' or 'multiclass'.")

        self.thr = cfg.metrics.threshold

        # ---- loss ----
        if self.task == "binary":
            self.criterion = DiceCELoss(
                sigmoid=True,
                lambda_dice=float(self.cfg.loss.lambda_dice),
                lambda_ce=float(self.cfg.loss.lambda_ce),
            )
        else:
            self.num_classes = int(cfg.data.num_classes)  # 包含背景
            self.include_background = bool(cfg.data.get("include_background", False))
            self.ignore_index = None if self.include_background else 0

            self.criterion = DiceCELoss(
                softmax=True,
                to_onehot_y=False,
                include_background=True,
                lambda_dice=float(self.cfg.loss.lambda_dice),
                lambda_ce=float(self.cfg.loss.lambda_ce),
            )

        # ---- metrics (separate for each stage) ----
        self.train_metrics = self._build_metrics()
        self.val_metrics = self._build_metrics()
        self.test_metrics = self._build_metrics()

    def _build_metrics(self) -> MetricCollection:
        if self.task == "binary":
            return MetricCollection({
                "Dice": BinaryF1Score(threshold=self.thr),
                "IoU": BinaryJaccardIndex(threshold=self.thr),
                "HD95": BinaryHD95(threshold=self.thr),
                "ASD": BinaryASD(threshold=self.thr),
                "Recall": BinaryRecall(threshold=self.thr),
                "Precision": BinaryPrecision(threshold=self.thr),
            })
        else:
            return MetricCollection({
                "Dice_macro": MulticlassF1Score(num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index),
                "IoU_macro": MulticlassJaccardIndex(num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index),
                "Dice_pc": MulticlassF1Score(num_classes=self.num_classes, average=None, ignore_index=self.ignore_index),
                "IoU_pc": MulticlassJaccardIndex(num_classes=self.num_classes, average=None, ignore_index=self.ignore_index),
            })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ---------- helpers ----------
    def _get_metrics(self, stage: str) -> MetricCollection:
        if stage == "train":
            return self.train_metrics
        if stage == "val":
            return self.val_metrics
        if stage == "test":
            return self.test_metrics
        raise ValueError(stage)

    def _resize_target_for_logits(self, target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if target.shape[-2:] == logits.shape[-2:]:
            return target
        return F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")

    def _compute_loss_from_outputs(self, outputs, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 兼容：
        # 1) 单输出 tensor
        # 2) 深监督 list / tuple，默认 outputs[0] 为主输出
        if isinstance(outputs, (list, tuple)):
            logits_list = list(outputs)
        else:
            logits_list = [outputs]

        main_logits = logits_list[0]

        if len(logits_list) == 1:
            loss = self.criterion(main_logits, target)
            return loss, main_logits

        # 深监督：主输出权重大，后续逐级衰减；并做归一化，避免整体 loss 尺度变化太大
        weights = [1.0] + [0.5 ** i for i in range(1, len(logits_list))]
        weight_sum = sum(weights)

        loss = 0.0
        for w, logits_i in zip(weights, logits_list):
            target_i = self._resize_target_for_logits(target, logits_i)
            loss = loss + w * self.criterion(logits_i, target_i)

        loss = loss / weight_sum
        return loss, main_logits

    def _compute_auxiliary_losses(self, target: torch.Tensor) -> dict[str, torch.Tensor]:
        auxiliary_fn = getattr(self.model, "auxiliary_losses", None)
        if auxiliary_fn is None:
            return {}
        losses = auxiliary_fn(target)
        if losses is None:
            return {}
        if not isinstance(losses, dict):
            raise TypeError("model.auxiliary_losses(target) must return a dict[str, Tensor].")
        return losses

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        x = batch["image"]
        y = batch["mask"]

        outputs = self(x)

        if self.task == "multiclass":
            y_oh = F.one_hot(y.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            target_for_loss = y_oh
        else:
            target_for_loss = y

        # ---------- loss ----------
        loss, logits = self._compute_loss_from_outputs(outputs, target_for_loss)
        auxiliary_losses = self._compute_auxiliary_losses(target_for_loss)
        if auxiliary_losses:
            loss = loss + sum(auxiliary_losses.values())

        if self.task == "multiclass":
            assert logits.shape[1] == self.num_classes, \
                f"multiclass requires logits channels C={self.num_classes}, but got {logits.shape}"

        # ---------- metrics update ----------
        metrics = self._get_metrics(stage)

        if self.task == "binary":
            # y: (B,1,H,W) float
            probs = torch.sigmoid(logits).detach()
            tgt = (y >= self.thr).int()  # (B,1,H,W) int
            metrics.update(probs, tgt)
        else:
            # y: (B,H,W) long
            pred = torch.argmax(logits, dim=1).detach()  # (B,H,W) long
            tgt = y.long()
            metrics.update(pred, tgt)

        for name, aux_loss in auxiliary_losses.items():
            self.log(
                f"{stage}/{name}",
                aux_loss.detach(),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=x.shape[0],
            )

        # loss log
        self.log(
            f"{stage}/loss",
            loss.detach(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=x.shape[0]
        )
        return loss

    # ---------- lightning hooks ----------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, stage="test")

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test")

    def _log_epoch_metrics(self, stage: str) -> None:
        metrics = self._get_metrics(stage)
        metric_results = metrics.compute()

        for metric_name, value in metric_results.items():
            # 关键：Dice_pc / IoU_pc 这种是向量，不能直接 log
            if isinstance(value, torch.Tensor) and value.numel() != 1:
                continue

            self.log(
                f"{stage}/{metric_name}",
                value,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True
            )

        if self.task == "multiclass":
            dice_vec = metric_results.get("Dice_pc", None)
            iou_vec = metric_results.get("IoU_pc", None)

            # ---- Dice: per-class + average(exclude background=0) ----
            if dice_vec is not None:
                dice_vec = torch.nan_to_num(dice_vec, nan=0.0)

                # 1..C-1 are organs (exclude background 0)
                organ_ids = list(range(1, int(dice_vec.numel())))
                if len(organ_ids) > 0:
                    # 计算每个器官的平均 Dice (不包含背景)
                    avg_dice = dice_vec[organ_ids].mean()

                    # log each organ dice
                    for c in organ_ids:
                        self.log(f"{stage}/Dice_class{c}", dice_vec[c],
                                 prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

                    # IMPORTANT: overwrite stage/Dice to Average for ckpt monitor val/Dice
                    self.log(f"{stage}/Dice", avg_dice,
                             prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

            # ---- IoU: per-class + average(exclude background=0) ----
            if iou_vec is not None:
                iou_vec = torch.nan_to_num(iou_vec, nan=0.0)

                organ_ids = list(range(1, int(iou_vec.numel())))
                if len(organ_ids) > 0:
                    # 计算每个器官的平均 IoU (不包含背景)
                    avg_iou = iou_vec[organ_ids].mean()

                    for c in organ_ids:
                        self.log(f"{stage}/IoU_class{c}", iou_vec[c],
                                 prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

                    # 记录总的平均 IoU
                    self.log(f"{stage}/IoU", avg_iou,
                             prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        metrics.reset()

    def configure_optimizers(self):
        opt_cfg = self.cfg.optim
        lr0 = float(opt_cfg.lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr0, weight_decay=opt_cfg.weight_decay)

        sched_name = opt_cfg.scheduler.lower()  # "poly" | "cosine" | "plateau"
        min_lr = float(opt_cfg.min_lr)

        if sched_name == "poly":
            def poly(step: int):
                total_steps = int(self.trainer.estimated_stepping_batches)
                if total_steps <= 1:
                    return 1.0
                step = min(step, total_steps)
                factor = (1.0 - step / total_steps) ** 0.9
                return max(min_lr / lr0, factor)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly)

        elif sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=min_lr,
            )

        elif sched_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.cfg.trainer.early_stopping.mode,
                factor=0.5,
                patience=10,
                threshold=1e-4,
                min_lr=min_lr,
            )

        else:
            raise ValueError(f"Unknown scheduler '{sched_name}'. Choose from: poly | cosine | plateau")

        print(f"[Optimization scheduler] {sched_name} lr0={lr0:.4f}, min_lr={min_lr:.6f}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.cfg.trainer.early_stopping.monitor if sched_name == "plateau" else None,
                "interval": "epoch" if sched_name in ("plateau", "cosine") else "step",
                "frequency": 1,
                "name": f"lr-{sched_name}"
            },
        }
