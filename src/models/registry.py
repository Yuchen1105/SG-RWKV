from __future__ import annotations

import importlib
import importlib.util
from typing import Callable, Dict

import torch.nn as nn

_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    def deco(fn: Callable[..., nn.Module]):
        if name in _MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' already registered")
        _MODEL_REGISTRY[name] = fn
        return fn

    return deco


def _ensure_registered(name: str) -> None:
    if name in _MODEL_REGISTRY:
        return

    candidates = {
        "HS3R_Net_V000": "src.models.HS3R_Net_V000",
        "HS3RNetV000": "src.models.HS3R_Net_V000",
        "U_RWKV": "src.models.u_rwkv",
        "u_rwkv_unet": "src.models.u_rwkv",
        "HS3R_Net_V005": "src.models.HS3R_Net_V005",
        "HS3R_Net_V005_Energy_NoApp": "src.models.HS3R_Net_V005_Energy_NoApp",
        "HS3R_Net_V005_Energy_NoSpa": "src.models.HS3R_Net_V005_Energy_NoSpa",
        "HS3R_Net_V005_Energy_NoBnd": "src.models.HS3R_Net_V005_Energy_NoBnd",
        "HS3R_Net_V005_Energy_DenseAssign": "src.models.HS3R_Net_V005_Energy_DenseAssign",
    }
    module_name = candidates.get(name)
    if module_name is None and name.startswith("HS3R_Net_V"):
        module_name = f"src.models.{name}"
    if module_name is not None and importlib.util.find_spec(module_name) is not None:
        importlib.import_module(module_name)


def build_model(name: str, **kwargs) -> nn.Module:
    _ensure_registered(name)
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    for name in (
        "HS3R_Net_V005",
        "HS3R_Net_V005_Energy_NoApp",
        "HS3R_Net_V005_Energy_NoSpa",
        "HS3R_Net_V005_Energy_NoBnd",
        "HS3R_Net_V005_Energy_DenseAssign",
        "U_RWKV",
    ):
        _ensure_registered(name)
    return sorted(_MODEL_REGISTRY)


def get_active_model(cfg):
    models = cfg.model.models
    active = cfg.model.get("active_model", 0)
    if isinstance(active, int):
        if active >= len(models):
            raise KeyError(f"active_model index {active} is out of range")
        return models[active]
    for spec in models:
        if spec.get("alias", spec.name) == active:
            return spec
    raise KeyError(f"Unknown active_model='{active}'")
