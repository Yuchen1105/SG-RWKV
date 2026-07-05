from __future__ import annotations

from .registry import register_model
from .HS3R_Net_SGRC_Base import HS3R_Net_SGRC_Base, SPV_CLASSES
from .HS3R_Net_SPV005_EnergyVariants import SPV005NoAppearance


SPV_CLASSES["SPV005NoAppearance"] = SPV005NoAppearance


class HS3R_Net_V005_Energy_NoApp(HS3R_Net_SGRC_Base):
    """HS3R-Net V005 energy ablation: SPV005 without appearance energy."""

    def __init__(self, **kwargs):
        kwargs.pop("spv_name", None)
        kwargs.pop("inject_mode", None)
        super().__init__(spv_name="SPV005NoAppearance", **kwargs)


@register_model("HS3R_Net_V005_Energy_NoApp")
def build_hs3r_net_v005_energy_noapp(**kwargs):
    return HS3R_Net_V005_Energy_NoApp(**kwargs)
