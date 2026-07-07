# src/data/datamodule.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import pytorch_lightning as pl

from .datasets import ImageMaskDataset
from .transforms import build_transforms


def _split_lengths(n: int, ratios: Dict[str, float]) -> Tuple[int, int, int]:
    r_train, r_val, r_test = ratios["train"], ratios["val"], ratios["test"]
    n_train = int(n * r_train)
    n_val = int(n * r_val)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def _stem(x: str) -> str:
    """'case001.png' -> 'case001' ; 'case001' -> 'case001' """
    b = os.path.basename(x)
    s, _ = os.path.splitext(b)
    return s


def _load_split_json(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"split_json must be a dict, got {type(obj)}")
    for k in ("train", "val", "test"):
        if k not in obj:
            raise ValueError(f"split_json missing key '{k}'. Required keys: train/val/test.")
        if not isinstance(obj[k], list):
            raise ValueError(f"split_json['{k}'] must be a list, got {type(obj[k])}")
    # normalize to stems
    return {k: [_stem(x) for x in obj[k]] for k in ("train", "val", "test")}


def _indices_from_ids(full_ds: Dataset, split_ids: Dict[str, List[str]]) -> Tuple[List[int], List[int], List[int]]:
    """
    Build indices for Subset based on sample ids.
    full_ds[i] must return dict with key 'id' as str (stem).
    """
    # build id -> index mapping once
    id2idx: Dict[str, int] = {}
    for i in range(len(full_ds)):
        item = full_ds[i]
        sid = str(item["id"])
        if sid in id2idx:
            raise ValueError(f"Duplicate sample id '{sid}' found in dataset. ids must be unique.")
        id2idx[sid] = i

    train_ids = split_ids["train"]
    val_ids = split_ids["val"]
    test_ids = split_ids["test"]

    # check duplicates across splits
    all_ids = train_ids + val_ids + test_ids
    if len(set(all_ids)) != len(all_ids):
        # find duplicates
        seen = set()
        dups = []
        for x in all_ids:
            if x in seen:
                dups.append(x)
            seen.add(x)
        raise ValueError(f"split_json contains duplicated ids across splits: {sorted(set(dups))[:50]}")

    # check existence
    unknown = [x for x in all_ids if x not in id2idx]
    if len(unknown) > 0:
        raise ValueError(
            f"split_json contains ids not found in dataset (showing up to 50): {unknown[:50]}\n"
            f"Hint: ids should match ImageMaskDataset returned 'id' (basename stem)."
        )

    train_idx = [id2idx[x] for x in train_ids]
    val_idx = [id2idx[x] for x in val_ids]
    test_idx = [id2idx[x] for x in test_ids]

    return train_idx, val_idx, test_idx


@dataclass
class DatasetSpec:
    name: str
    root: str


class MultiDatasetDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None
        self.activate_dataset_name: Optional[str] = None
        self.activate_dataset_root: Optional[str] = None

    def _persistent_workers_flag(self, num_workers: int) -> bool:
        cfg_value = self.cfg.data.get("persistent_workers", None)
        if cfg_value is None:
            return num_workers > 0
        return bool(cfg_value) and num_workers > 0

    def _select_dataset_spec(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select exactly ONE dataset spec from cfg.data.datasets.

        activate_dataset supports:
            - int index
            - str name (match ds_spec["name"])
        """
        dcfg = self.cfg.data
        activate: Union[int, str] = dcfg.get("activate_dataset", 0)

        if isinstance(activate, int):
            if activate < 0 or activate >= len(datasets):
                raise IndexError(f"activate_dataset={activate} out of range [0, {len(datasets)-1}]")
            return datasets[activate]

        if isinstance(activate, str):
            for ds in datasets:
                if ds.get("name", None) == activate:
                    return ds
            available = [d.get("name", "<no-name>") for d in datasets]
            raise ValueError(f"activate_dataset='{activate}' not found. Available names: {available}")

        raise TypeError("cfg['data']['activate_dataset'] must be int (index) or str (dataset name).")

    def setup(self, stage: Optional[str] = None) -> None:
        dcfg = self.cfg["data"]
        datasets: List[Dict[str, Any]] = dcfg["datasets"]
        if len(datasets) == 0:
            raise RuntimeError("No datasets specified in cfg['data']['datasets'].")

        ds_spec = self._select_dataset_spec(datasets)
        gen = torch.Generator().manual_seed(int(self.cfg.get("seed", 42)))

        root = ds_spec["root"]
        name = ds_spec.get("name", "dataset")
        images_dir = f"{root}/{dcfg['image_dirname']}"
        masks_dir = f"{root}/{dcfg['mask_dirname']}"

        self.activate_dataset_name = name
        self.activate_dataset_root = root

        tf_train = build_transforms(dcfg.input_size, dcfg.augment, stage="train")
        tf_eval = build_transforms(dcfg.input_size, None, stage="eval")

        full_trainable = ImageMaskDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            image_channels=int(dcfg.image_channels),
            transform=None,
            mask_threshold=float(dcfg.mask_threshold),
            task=str(dcfg.task),
            num_classes=int(dcfg.num_classes),
        )

        # ---- split: json (manual) OR random ----
        split_json = ds_spec["split_json"]
        if split_json is not None and str(split_json).strip() != "":
            split_path = str(split_json)
            if not os.path.isabs(split_path):
                cand = os.path.join(root, split_path)
                split_path = cand if os.path.exists(cand) else split_path
            if not os.path.exists(split_path):
                raise FileNotFoundError(f"split_json not found: {split_path}")

            split_ids = _load_split_json(split_path)
            train_idx, val_idx, test_idx = _indices_from_ids(full_trainable, split_ids)

            train_ds = Subset(full_trainable, train_idx)
            val_ds = Subset(full_trainable, val_idx)
            test_ds = Subset(full_trainable, test_idx)

            n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)
            n = n_train + n_val + n_test
            print(f"[DataModule] Using manual split_json: {split_path}")
        else:
            n = len(full_trainable)
            n_train, n_val, n_test = _split_lengths(n, dcfg["split"])
            train_ds, val_ds, test_ds = random_split(full_trainable, [n_train, n_val, n_test], generator=gen)

        # attach transforms
        self.train_ds = _TransformedDataset(train_ds, tf_train)
        self.val_ds = _TransformedDataset(val_ds, tf_eval)
        self.test_ds = _TransformedDataset(test_ds, tf_eval)

        print(f"[DataModule] {name}: {n} samples, {n_train} train, {n_val} val, {n_test} test")

    def train_dataloader(self) -> DataLoader:
        dcfg = self.cfg.data
        num_workers = int(dcfg.num_workers)
        return DataLoader(
            self.train_ds,
            batch_size=int(dcfg.batch_size),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=bool(dcfg.pin_memory),
            drop_last=False,
            persistent_workers=self._persistent_workers_flag(num_workers),
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None or len(self.val_ds) == 0:       ############Synapse
            return None
        dcfg = self.cfg.data
        num_workers = int(dcfg.num_workers)
        return DataLoader(
            self.val_ds,
            batch_size=int(dcfg.batch_size),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(dcfg.pin_memory),
            drop_last=False,
            persistent_workers=self._persistent_workers_flag(num_workers),
        )
    # def val_dataloader(self) -> DataLoader:
    #     dcfg = self.cfg.data
    #
    #     # 没有验证集：返回一个“空”的可迭代 dataloader（绝对不要 return None）
    #     if self.val_ds is None or len(self.val_ds) == 0:
    #         empty = TensorDataset(torch.empty(0, dtype=torch.uint8))  # 0 个样本
    #         return DataLoader(
    #             empty,
    #             batch_size=1,
    #             shuffle=False,
    #             num_workers=0,
    #             pin_memory=False,
    #         )
    def test_dataloader(self) -> DataLoader:
        dcfg = self.cfg.data
        num_workers = int(dcfg.num_workers)
        return DataLoader(
            self.test_ds,
            batch_size=int(dcfg.batch_size),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(dcfg.pin_memory),
            drop_last=False,
            persistent_workers=self._persistent_workers_flag(num_workers),
        )


class _TransformedDataset(Dataset):
    def __init__(self, base: Dataset, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]  # dict(image, mask, id)
        img, msk = item["image"], item["mask"]
        img, msk = self.transform(img, msk)
        item["image"], item["mask"] = img, msk
        return item
