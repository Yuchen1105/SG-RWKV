# src/data/transforms.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _is_multiclass_mask(mask: torch.Tensor) -> bool:
    return mask.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8)


def _ensure_mask_shape(mask: torch.Tensor) -> torch.Tensor:
    if _is_multiclass_mask(mask):
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        return mask
    else:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return mask


@dataclass
class JointAugment:
    """Apply the same random transforms to image and mask.

    Works for:
      - binary mask: (1,H,W)
      - multiclass label map: (H,W)
    """
    hflip: bool = True
    vflip: bool = True
    rot90: bool = True

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        # image: (C,H,W)
        mask = _ensure_mask_shape(mask)

        if self.hflip and torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])  # W
            mask = torch.flip(mask, dims=[-1])   # last dim is W

        if self.vflip and torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[1])  # H
            mask = torch.flip(mask, dims=[-2])   # second last dim is H

        if self.rot90:
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                image = torch.rot90(image, k, dims=[1, 2])
                mask = torch.rot90(mask, k, dims=[-2, -1])

        return image, mask


@dataclass
class Resize:
    size: Optional[Tuple[int, int]] = None  # (H, W)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if self.size is None:
            return image, mask

        mask = _ensure_mask_shape(mask)
        h, w = self.size

        # image: (C,H,W) -> (1,C,H,W)
        image_b = image.unsqueeze(0)
        image_b = F.interpolate(image_b, size=(h, w), mode="bilinear", align_corners=False)
        image = image_b.squeeze(0)

        # mask:
        # - binary: (1,H,W) float -> (1,1,H,W) -> nearest -> (1,H,W) float
        # - multiclass: (H,W) long(label) -> float -> (1,1,H,W) -> nearest -> long -> (H,W)
        if _is_multiclass_mask(mask):
            # (H,W) -> (1,1,H,W)
            mask_f = mask.to(torch.float32).unsqueeze(0).unsqueeze(0)
            mask_f = F.interpolate(mask_f, size=(h, w), mode="nearest")
            mask = mask_f.squeeze(0).squeeze(0).to(torch.long)  # back to (H,W) long
        else:
            # binary float mask, ensure (1,H,W)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask_b = mask.unsqueeze(0)  # (1,1,H,W)
            mask_b = F.interpolate(mask_b, size=(h, w), mode="nearest")
            mask = mask_b.squeeze(0)  # (1,H,W) float

        return image, mask


class Compose:
    def __init__(self, *ops):
        self.ops = [op for op in ops if op is not None]

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        for op in self.ops:
            image, mask = op(image, mask)
        return image, mask


def build_transforms(input_size=None, augment=None, stage: str = "train"):
    resize = Resize(size=tuple(input_size) if input_size is not None else None)
    if stage == "train":
        aug = JointAugment(**augment) if augment is not None else None
        return Compose(resize, aug)
    return Compose(resize)