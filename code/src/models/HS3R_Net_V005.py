from __future__ import annotations

from .registry import register_model
from .HS3R_Net_SGRC_Base import HS3R_Net_SGRC_Base


class HS3R_Net_V005(HS3R_Net_SGRC_Base):
    """SGRC model using SPV005: differentiable SEEDS-style energy sampling."""

    def __init__(self, **kwargs):
        kwargs.pop("spv_name", None)
        kwargs.pop("inject_mode", None)
        super().__init__(spv_name="SPV005", **kwargs)


@register_model("HS3R_Net_V005")
def build_hs3r_net_v005(**kwargs):
    return HS3R_Net_V005(**kwargs)
