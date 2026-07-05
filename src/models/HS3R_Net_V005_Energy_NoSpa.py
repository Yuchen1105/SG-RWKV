from __future__ import annotations

from .registry import register_model
from .HS3R_Net_SGRC_Base import HS3R_Net_SGRC_Base, SPV_CLASSES
from .HS3R_Net_SPV005_EnergyVariants import SPV005NoSpatial


SPV_CLASSES["SPV005NoSpatial"] = SPV005NoSpatial


class HS3R_Net_V005_Energy_NoSpa(HS3R_Net_SGRC_Base):
    """HS3R-Net V005 energy ablation: SPV005 without spatial compactness."""

    def __init__(self, **kwargs):
        kwargs.pop("spv_name", None)
        kwargs.pop("inject_mode", None)
        super().__init__(spv_name="SPV005NoSpatial", **kwargs)


@register_model("HS3R_Net_V005_Energy_NoSpa")
def build_hs3r_net_v005_energy_nospa(**kwargs):
    return HS3R_Net_V005_Energy_NoSpa(**kwargs)
