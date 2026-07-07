from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import MultiDatasetDataModule
from src.lightning.seg_module import SegLitModule
from src.models.registry import build_model, get_active_model


def parse_devices(value: str):
    if "," in value:
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    if value.isdigit():
        return int(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HS3R-Net.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Lightning checkpoint path.")
    parser.add_argument("--dataset", type=str, default=None, help="Override cfg.data.activate_dataset.")
    parser.add_argument("--data-root", type=str, default=None, help="Optional dataset root override.")
    parser.add_argument("--devices", type=str, default=None, help="Lightning devices argument.")
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA WKV and evaluate on CPU.")
    return parser.parse_args()


def prepare_config(args: argparse.Namespace):
    cfg = OmegaConf.load(args.config)
    if args.dataset is not None:
        cfg.data.activate_dataset = args.dataset
    if args.data_root is not None:
        active = cfg.data.activate_dataset
        for ds in cfg.data.datasets:
            if ds.name == active:
                ds.root = args.data_root
                break
    if args.devices is not None:
        cfg.trainer.devices = parse_devices(args.devices)
    if args.cpu:
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1
    return cfg


def main() -> None:
    args = parse_args()
    cfg = prepare_config(args)
    pl.seed_everything(int(cfg.seed), workers=True)

    spec = get_active_model(cfg)
    kwargs = dict(spec.get("kwargs", {}))
    if kwargs.get("wkv_cuda_dir", None) is None:
        kwargs["wkv_cuda_dir"] = str(REPO_ROOT / "src" / "models")
    if args.cpu:
        kwargs["enable_wkv_cuda"] = False

    model = build_model(spec.name, **kwargs)
    lit = SegLitModule.load_from_checkpoint(args.checkpoint, model=model, cfg=cfg, strict=False)
    datamodule = MultiDatasetDataModule(cfg)

    trainer = pl.Trainer(
        accelerator=str(cfg.trainer.accelerator),
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=False,
    )
    trainer.test(lit, datamodule=datamodule)


if __name__ == "__main__":
    main()
