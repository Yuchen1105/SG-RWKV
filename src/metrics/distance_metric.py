# src/metrics/distance_metric.py
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage
from torchmetrics import Metric

def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    """Compute boundary pixels of a 2D binary mask."""
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)), iterations=1)
    boundary = mask ^ eroded
    return boundary


def _surface_distances(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Optional[Sequence[float]] = None,
    empty_strategy: str = "diag",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute directed surface distances (pred->gt, gt->pred).

    Returns:
        d1: distances from pred surface to gt surface
        d2: distances from gt surface to pred surface
    """
    assert pred.ndim == 2 and gt.ndim == 2
    spacing = spacing if spacing is not None else (1.0, 1.0)

    spred = _binary_boundary(pred)
    sgt = _binary_boundary(gt)

    # both empty
    if spred.sum() == 0 and sgt.sum() == 0:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)

    # one empty
    if spred.sum() == 0 or sgt.sum() == 0:
        if empty_strategy == "inf":
            inf = np.array([np.inf], dtype=np.float32)
            return inf, inf
        # diag strategy
        h, w = pred.shape
        diag = float(np.sqrt((h * spacing[0]) ** 2 + (w * spacing[1]) ** 2))
        val = np.array([diag], dtype=np.float32)
        return val, val

    # distance transform of the complement of the surface (False=surface pixels -> distance 0)
    dt_gt = ndimage.distance_transform_edt(~sgt, sampling=spacing)
    dt_pr = ndimage.distance_transform_edt(~spred, sampling=spacing)

    d1 = dt_gt[spred].astype(np.float32)
    d2 = dt_pr[sgt].astype(np.float32)
    return d1, d2


class BinaryHD95(Metric):
    """HD95 for binary segmentation using TorchMetrics Metric API.

    - Inputs:
        preds: probabilities/logits tensor (N, 1, H, W) or (N, H, W)
        target: binary tensor (N, 1, H, W) or (N, H, W)
    - Output:
        mean HD95 over batch updates
    """

    full_state_update = False

    def __init__(
        self,
        threshold: float = 0.5,
        spacing: Optional[Sequence[float]] = None,
        empty_strategy: str = "diag",
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.threshold = float(threshold)
        self.spacing = spacing
        self.empty_strategy = empty_strategy

        self.add_state("sum_hd95", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # move to cpu for scipy ops
        if preds.ndim == 4:
            preds = preds[:, 0]
        if target.ndim == 4:
            target = target[:, 0]

        # accept logits or probs
        if preds.dtype.is_floating_point:
            # if looks like logits, sigmoid; otherwise threshold directly is ok
            probs = torch.sigmoid(preds) if preds.min() < 0 or preds.max() > 1 else preds
        else:
            probs = preds.float()

        pred_bin = (probs >= self.threshold).to(torch.uint8)
        tgt_bin = (target >= 0.5).to(torch.uint8)

        for i in range(pred_bin.shape[0]):
            p = pred_bin[i].detach().cpu().numpy().astype(bool)
            g = tgt_bin[i].detach().cpu().numpy().astype(bool)
            d1, d2 = _surface_distances(p, g, spacing=self.spacing, empty_strategy=self.empty_strategy)
            all_d = np.concatenate([d1, d2], axis=0)
            hd95 = float(np.percentile(all_d, 95))
            self.sum_hd95 += torch.tensor(hd95, device=self.sum_hd95.device)
            self.count += 1

    def compute(self) -> torch.Tensor:
        return self.sum_hd95 / torch.clamp(self.count, min=1)


class BinaryASD(Metric):
    """Average Symmetric Surface Distance (ASD/ASSD) for binary segmentation."""

    full_state_update = False

    def __init__(
        self,
        threshold: float = 0.5,
        spacing: Optional[Sequence[float]] = None,
        empty_strategy: str = "diag",
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = float(threshold)
        self.spacing = spacing
        self.empty_strategy = empty_strategy

        self.add_state("sum_asd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.ndim == 4:
            preds = preds[:, 0]
        if target.ndim == 4:
            target = target[:, 0]

        if preds.dtype.is_floating_point:
            probs = torch.sigmoid(preds) if preds.min() < 0 or preds.max() > 1 else preds
        else:
            probs = preds.float()

        pred_bin = (probs >= self.threshold).to(torch.uint8)
        tgt_bin = (target >= 0.5).to(torch.uint8)

        for i in range(pred_bin.shape[0]):
            p = pred_bin[i].detach().cpu().numpy().astype(bool)
            g = tgt_bin[i].detach().cpu().numpy().astype(bool)
            d1, d2 = _surface_distances(p, g, spacing=self.spacing, empty_strategy=self.empty_strategy)
            asd = 0.5 * (float(d1.mean()) + float(d2.mean()))
            self.sum_asd += torch.tensor(asd, device=self.sum_asd.device)
            self.count += 1

    def compute(self) -> torch.Tensor:
        return self.sum_asd / torch.clamp(self.count, min=1)