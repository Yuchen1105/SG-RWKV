from __future__ import annotations

import argparse
import csv
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
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint on multiple target datasets.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--targets", nargs="+", required=True, help="Target dataset names from cfg.data.datasets.")
    parser.add_argument("--output", type=str, default="cross_dataset_results.csv")
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def build_lit_module(cfg, checkpoint: str, cpu: bool) -> SegLitModule:
    spec = get_active_model(cfg)
    kwargs = dict(spec.get("kwargs", {}))
    if kwargs.get("wkv_cuda_dir", None) is None:
        kwargs["wkv_cuda_dir"] = str(REPO_ROOT / "src" / "models")
    if cpu:
        kwargs["enable_wkv_cuda"] = False
    model = build_model(spec.name, **kwargs)
    return SegLitModule.load_from_checkpoint(checkpoint, model=model, cfg=cfg, strict=False)


def main() -> None:
    args = parse_args()
    base_cfg = OmegaConf.load(args.config)
    if args.devices is not None:
        base_cfg.trainer.devices = parse_devices(args.devices)
    if args.cpu:
        base_cfg.trainer.accelerator = "cpu"
        base_cfg.trainer.devices = 1

    rows = []
    for target in args.targets:
        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        cfg.data.activate_dataset = target
        pl.seed_everything(int(cfg.seed), workers=True)
        lit = build_lit_module(cfg, args.checkpoint, args.cpu)
        datamodule = MultiDatasetDataModule(cfg)
        trainer = pl.Trainer(
            accelerator=str(cfg.trainer.accelerator),
            devices=cfg.trainer.devices,
            precision=cfg.trainer.precision,
            logger=False,
        )
        result = trainer.test(lit, datamodule=datamodule)[0]
        rows.append({
            "target_dataset": target,
            "Dice": result.get("test/Dice"),
            "IoU": result.get("test/IoU"),
            "Recall": result.get("test/Recall"),
            "Precision": result.get("test/Precision"),
            "HD95": result.get("test/HD95"),
            "ASD": result.get("test/ASD"),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved cross-dataset results to {output}")


if __name__ == "__main__":
    main()
