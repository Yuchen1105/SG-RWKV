from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

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
    parser = argparse.ArgumentParser(description="Train HS3R-Net.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--dataset", type=str, default=None, help="Override cfg.data.activate_dataset.")
    parser.add_argument("--data-root", type=str, default=None, help="Optional dataset root override.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for logs/checkpoints.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs.")
    parser.add_argument("--devices", type=str, default=None, help="Lightning devices argument, e.g. 1 or 0,1.")
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA WKV and run on CPU.")
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
    if args.output_dir is not None:
        cfg.logging.save_dir = args.output_dir
    if args.max_epochs is not None:
        cfg.trainer.max_epochs = args.max_epochs
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
    lit = SegLitModule(model=model, cfg=cfg)
    datamodule = MultiDatasetDataModule(cfg)

    save_dir = Path(str(cfg.logging.save_dir))
    logger = TensorBoardLogger(save_dir=str(save_dir), name=str(cfg.logging.experiment_name))
    checkpoint = ModelCheckpoint(
        monitor=str(cfg.trainer.early_stopping.monitor),
        mode=str(cfg.trainer.early_stopping.mode),
        save_top_k=1,
        save_last=True,
        filename="epoch{epoch:03d}",
        auto_insert_metric_name=False,
    )
    early_stop = EarlyStopping(
        monitor=str(cfg.trainer.early_stopping.monitor),
        mode=str(cfg.trainer.early_stopping.mode),
        patience=int(cfg.trainer.early_stopping.patience),
    )

    trainer = pl.Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=str(cfg.trainer.accelerator),
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        logger=logger,
        callbacks=[checkpoint, early_stop, LearningRateMonitor(logging_interval="epoch")],
    )
    trainer.fit(lit, datamodule=datamodule)
    trainer.test(lit, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
