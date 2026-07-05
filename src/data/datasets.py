# src/data/datasets.py
from __future__ import annotations

import os
from dataclasses import dataclass

from typing import Callable, Optional, Sequence, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile, UnidentifiedImageError

from ..utils.misc import list_image_files, stem


IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy")
ImageFile.LOAD_TRUNCATED_IMAGES = True

def _normalize_mask_stem(s: str) -> str:
    # 兼容：case001_tumor.png / ISIC_xxx_segmentation.png / case022_other1.png
    for suf in ("_segmentation", "_tumor", "_mask", "-segmentation", "-tumor", "-mask"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s



import re

def _mask_base_id(mask_stem: str) -> str:
    """
    把 mask 文件名 stem 归一到 image stem：

    例子：
      case210_tumor        -> case210
      case210_mask         -> case210
      case210_segmentation -> case210
      case140_other3       -> case140
      case140-other12      -> case140
    """
    s = mask_stem.lower()
    s = re.sub(r"\s*\(\d+\)$", "", s)  # 去掉 " (1)" 这种
    s = re.sub(r"([_-](segmentation|mask|tumor|anno))$", "", s)
    # 2) 去掉 otherN（末尾一次）
    s = re.sub(r"([_-]other\d+)$", "", s)

    return s



def _load_2d(path: str, image_channels: int) -> np.ndarray:
    """Load image or mask as 2D numpy array.

    - For image: returns float32 array (H, W) or (H, W, C)
    - For mask : returns float32 array (H, W)
    """
    if path.lower().endswith(".npy"):
        arr = np.load(path)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            # (C,H,W) -> (H,W,C)
            arr = np.transpose(arr, (1, 2, 0))
        return arr.astype(np.float32)

    with Image.open(path) as img:
        img.load()
        arr = np.array(img)
    return arr.astype(np.float32)


def _probe_2d(path: str) -> tuple[bool, str]:
    if path.lower().endswith(".npy"):
        try:
            arr = np.load(path, mmap_mode="r")
            _ = arr.shape
            return True, ""
        except Exception as e:
            return False, str(e)

    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        return True, ""
    except (OSError, ValueError, UnidentifiedImageError) as e:
        return False, str(e)


def _ensure_channels(arr: np.ndarray, image_channels: int) -> np.ndarray:
    if arr.ndim == 2:
        if image_channels == 1:
            return arr[..., None]  # (H,W,1)
        # expand to 3 channels
        return np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim == 3:
        if arr.shape[-1] == image_channels:
            return arr
        if image_channels == 1:
            # convert RGB -> gray by simple average
            return arr.mean(axis=-1, keepdims=True)
        if arr.shape[-1] == 1 and image_channels == 3:
            return np.repeat(arr, 3, axis=-1)
    raise ValueError(f"Unsupported image shape {arr.shape} for image_channels={image_channels}")


@dataclass
class SamplePaths:
    image_path: str
    mask_paths: Sequence[str]



class ImageMaskDataset(Dataset):
    """Generic 2D segmentation dataset supporting binary and multiclass label maps.

    Pairing rule: match by basename *stem* (without extension).

    Modes:
      - task="binary": returns mask float tensor (1,H,W) in {0,1}
      - task="multiclass": returns mask long tensor (H,W) with values in [0, num_classes-1]
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_channels: int = 1,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        mask_threshold: float = 0.5,
        task: str = "binary",
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_channels = int(image_channels)
        self.transform = transform
        self.mask_threshold = float(mask_threshold)

        self.task = str(task).lower()
        if self.task not in ("binary", "multiclass"):
            raise ValueError(f"Unknown task='{task}'. Use 'binary' or 'multiclass'.")
        self.num_classes = int(num_classes) if num_classes is not None else None
        if self.task == "multiclass" and (self.num_classes is None or self.num_classes < 2):
            raise ValueError("For task='multiclass', num_classes must be provided and >= 2.")

        img_files = list_image_files(images_dir, IMG_EXTS)
        msk_files = list_image_files(masks_dir, IMG_EXTS)

        # images: key = image stem（建议 lower 防大小写坑）
        img_map = {stem(p).lower(): p for p in img_files}

        # masks: key = base id, value = list of mask paths
        msk_map: dict[str, List[str]] = {}
        for p in msk_files:
            s = _mask_base_id(stem(p))
            msk_map.setdefault(s, []).append(p)

        keys = sorted(set(img_map.keys()) & set(msk_map.keys()))
        print(f"[ImageMaskDataset] images={len(img_map)} masks={len(msk_files)} matched={len(keys)}")

        if len(keys) == 0:
            raise RuntimeError(
                f"No matched image-mask pairs found. images_dir={images_dir}, masks_dir={masks_dir}\n"
                "Hint: make sure basenames match, e.g. case001.png <-> case001.png"
            )

        valid_samples: list[SamplePaths] = []
        skipped_samples: list[tuple[str, str]] = []
        for k in keys:
            sample = SamplePaths(img_map[k], msk_map[k])
            ok, msg = _probe_2d(sample.image_path)
            if not ok:
                skipped_samples.append((sample.image_path, msg))
                continue

            broken_mask = None
            for mp in sample.mask_paths:
                ok, msg = _probe_2d(mp)
                if not ok:
                    broken_mask = (mp, msg)
                    break
            if broken_mask is not None:
                skipped_samples.append(broken_mask)
                continue

            valid_samples.append(sample)

        if skipped_samples:
            print(f"[ImageMaskDataset] skipped_corrupt={len(skipped_samples)}")
            for bad_path, reason in skipped_samples[:10]:
                print(f"[ImageMaskDataset] skip: {bad_path} | {reason}")

        if len(valid_samples) == 0:
            raise RuntimeError("All matched image-mask pairs are unreadable after validation.")

        self.samples = valid_samples

    def __len__(self) -> int:
        return len(self.samples)

    # def _postprocess_mask_binary(self, msk: np.ndarray) -> torch.Tensor:
    #     # msk: (H,W) float in [0,1] or uint8-like
    #     if msk.max() > 1.5:
    #         msk = msk / 255.0
    #     msk_t = torch.from_numpy(msk[None, ...]).float()  # (1,H,W)
    #     msk_t = (msk_t >= self.mask_threshold).float()
    #     return msk_t
    def _postprocess_mask_binary(self, msk: np.ndarray) -> torch.Tensor:
        # msk: (H,W) or (H,W,1/3)
        m = msk
        if m.ndim == 3:
            m = m[..., 0]

        # 先转 float 方便统计
        mf = m.astype(np.float32)
        mx = float(mf.max()) if mf.size else 0.0

        # 取一个稀疏采样的 unique（避免太慢）
        u = np.unique(mf[::8, ::8])
        u_n = int(u.size)

        # --- 情况1：标签图（最典型：0/1/2/...）---
        # 经验判断：最大值不大且 unique 很少 -> 当作 label map
        if mx <= 20 and u_n <= 64:
            bin_m = (mf > 0).astype(np.float32)
            return torch.from_numpy(bin_m[None, ...]).float()  # (1,H,W)

        # --- 情况2：灰度概率/灰度mask（0..255 或 0..1 连续）---
        # 0..255
        if mx > 1.5:
            mf = mf / 255.0

        msk_t = torch.from_numpy(mf[None, ...]).float()
        msk_t = (msk_t >= self.mask_threshold).float()
        return msk_t

    def _postprocess_mask_multiclass(self, msk: np.ndarray) -> torch.Tensor:
        C = self.num_classes  # type: ignore
        m = msk
        if m.dtype != np.int32 and m.dtype != np.int64:
            # float 或 uint8 都先转 float 方便判断
            m = m.astype(np.float32)

        mmax = float(m.max()) if m.size > 0 else 0.0
        if mmax <= (C - 1) + 1e-6:
            lab = np.rint(m).astype(np.int64)
        else:
            # 假设是 0..255 灰度编码
            lab = np.rint((m / 255.0) * (C - 1)).astype(np.int64)

        lab = np.clip(lab, 0, C - 1).astype(np.int64)
        return torch.from_numpy(lab).long()  # (H,W)

    def __getitem__(self, idx: int) -> dict:
        sp = self.samples[idx]
        img = _load_2d(sp.image_path, self.image_channels)

        # ---- load & union multiple masks ----
        msks = []
        for mp in sp.mask_paths:
            m = _load_2d(mp, 1)
            if m.ndim == 3:
                m = m[..., 0]
            msks.append(m)

        msk = msks[0] if len(msks) == 1 else np.maximum.reduce(msks)

        img = _ensure_channels(img, self.image_channels)  # (H,W,C)

        # normalize image to [0,1] if uint8-like
        if img.max() > 1.5:
            img = img / 255.0

        # to tensor: image (C,H,W)
        img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

        # mask tensor depending on task
        if self.task == "binary":
            msk_t = self._postprocess_mask_binary(msk)          # (1,H,W) float
        else:
            msk_t = self._postprocess_mask_multiclass(msk)      # (H,W) long

        if self.transform is not None:
            img_t, msk_t = self.transform(img_t, msk_t)

        return dict(
            image=img_t,
            mask=msk_t,
            id=os.path.splitext(os.path.basename(sp.image_path))[0],
        )
