from __future__ import annotations

from .registry import register_model
from .HS3R_Net_SGRC_Base import HS3R_Net_SGRC_Base, SPV_CLASSES
from .HS3R_Net_SPV005_EnergyVariants import SPV005DenseAssignment


SPV_CLASSES["SPV005DenseAssignment"] = SPV005DenseAssignment


class HS3R_Net_V005_Energy_DenseAssign(HS3R_Net_SGRC_Base):
    """HS3R-Net V005 energy ablation: SPV005 dense assignment."""

    def __init__(self, **kwargs):
        kwargs.pop("spv_name", None)
        kwargs.pop("inject_mode", None)
        super().__init__(spv_name="SPV005DenseAssignment", **kwargs)


@register_model("HS3R_Net_V005_Energy_DenseAssign")
def build_hs3r_net_v005_energy_denseassign(**kwargs):
    return HS3R_Net_V005_Energy_DenseAssign(**kwargs)
