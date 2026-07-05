from __future__ import annotations

import math
import inspect
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .HS3R_Net_Block import SPV001, SPV002, SPV003, SPV004, SPV005, SPV006, SPV007
from .HS3R_Net_V000 import (
    BiRWKV2D,
    ConvBNAct,
    FusionConv,
    RUN_WKV,
    SpectralMixer,
    T_MAX_DEFAULT,
    UpConv,
    WKV,
    _WKVBackend,
    q_shift,
)


SPV_CLASSES = {
    "SPV001": SPV001,
    "SPV002": SPV002,
    "SPV003": SPV003,
    "SPV004": SPV004,
    "SPV005": SPV005,
    "SPV006": SPV006,
    "SPV007": SPV007,
}


def _as_tuple4(value, default: int = 64) -> tuple[int, int, int, int]:
    if value is None:
        return (default, default, default, default)
    if isinstance(value, int):
        return (value, value, value, value)
    vals = tuple(int(v) for v in value)
    if len(vals) != 4:
        raise ValueError(f"Expected 4 values, got {vals}.")
    return vals


class _SPVAdapter(nn.Module):
    def __init__(
        self,
        channels: int,
        spv_name: str,
        n_spixels: int = 64,
        max_spatial: int = 128,
        assignment_topk: int = 9,
        spv_kwargs: Optional[dict[str, object]] = None,
    ) -> None:
        super().__init__()
        if spv_name not in SPV_CLASSES:
            raise KeyError(f"Unknown spv_name={spv_name}. Available: {list(SPV_CLASSES)}")
        self.max_spatial = int(max_spatial)
        spv_cls = SPV_CLASSES[spv_name]
        kwargs = {"in_channels": channels, "embed_channels": channels, "n_spixels": int(n_spixels)}
        if "assignment_topk" in inspect.signature(spv_cls.__init__).parameters:
            kwargs["assignment_topk"] = int(assignment_topk)
        if spv_kwargs is not None:
            kwargs.update(spv_kwargs)
        self.spv = spv_cls(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        scale = min(1.0, float(self.max_spatial) / float(max(h, w)))
        if scale < 1.0:
            x_in = F.interpolate(
                x,
                size=(max(1, int(round(h * scale))), max(1, int(round(w * scale)))),
                mode="bilinear",
                align_corners=False,
            )
        else:
            x_in = x

        sp_feat = self.spv(x_in).pixel_features
        if sp_feat.shape[-2:] != (h, w):
            sp_feat = F.interpolate(sp_feat, size=(h, w), mode="bilinear", align_corners=False)
        return sp_feat


class _SPGateSource(nn.Module):
    def __init__(
        self,
        channels: int,
        spv_name: str,
        n_spixels: int,
        max_spatial: int,
        assignment_topk: int,
        spv_kwargs: Optional[dict[str, object]] = None,
    ) -> None:
        super().__init__()
        self.spv = _SPVAdapter(
            channels,
            spv_name,
            n_spixels=n_spixels,
            max_spatial=max_spatial,
            assignment_topk=assignment_topk,
            spv_kwargs=spv_kwargs,
        )
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=True), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(self.spv(x))


class _SGRCSpatialInteractionMix(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_layer: int,
        layer_id: int,
        shift_pixel: int = 1,
        channel_gamma: float = 1 / 4,
        key_norm: bool = True,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        attn_sz = n_embd

        with torch.no_grad():
            ratio_0_to_1 = self.layer_id / (self.n_layer - 1 + 1e-6)
            ratio_1_to_almost0 = 1.0 - (self.layer_id / (self.n_layer + 1e-6))

            decay_speed = torch.ones(self.n_embd)
            for h in range(self.n_embd):
                decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1 + 1e-6)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.spatial_decay = nn.Parameter(decay_speed)

            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)], dtype=torch.float32) * 0.5
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)

            x = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                x[0, 0, i] = i / self.n_embd
            self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.shift_pixel = int(shift_pixel)
        self.channel_gamma = float(channel_gamma)
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.key_norm = nn.LayerNorm(attn_sz) if key_norm else None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

    def forward(self, x: torch.Tensor, resolution: Tuple[int, int], sp_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, c = x.shape
        if self.shift_pixel > 0:
            xx = q_shift(x, self.shift_pixel, self.channel_gamma, resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk, xv, xr = x, x, x

        k = self.key(xk)
        v = self.value(xv)
        sr = torch.sigmoid(self.receptance(xr))
        if sp_gate is not None:
            sr = sr * sp_gate

        w = (self.spatial_decay / t).to(device=x.device, dtype=x.dtype)
        u = (self.spatial_first / t).to(device=x.device, dtype=x.dtype)
        rwkv = RUN_WKV(b, t, c, w, u, k, v)
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        return self.output(sr * rwkv)


class _SGRCRWKVBlock2D(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_layer: int,
        layer_id: int,
        hidden_rate: int = 4,
        key_norm: bool = False,
        ffn_first: bool = False,
        shift_pixel: int = 1,
        channel_gamma: float = 1 / 4,
    ) -> None:
        super().__init__()
        self.ffn_first = ffn_first
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.att = _SGRCSpatialInteractionMix(n_embd, n_layer, layer_id, shift_pixel, channel_gamma, key_norm)
        self.ffn = SpectralMixer(n_embd, n_layer, layer_id, shift_pixel, channel_gamma, hidden_rate, key_norm)
        self.gamma1 = nn.Parameter(torch.ones((n_embd,)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd,)), requires_grad=True)

    def forward(self, x: torch.Tensor, sp_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, h, w = x.shape
        tok = rearrange(x, "b c h w -> b (h w) c")
        gate_tok = None if sp_gate is None else rearrange(sp_gate, "b c h w -> b (h w) c")
        resolution = (h, w)

        if self.ffn_first:
            tok = tok + self.gamma2 * self.ffn(self.ln2(tok), resolution)
            tok = tok + self.gamma1 * self.att(self.ln1(tok), resolution, gate_tok)
        else:
            tok = tok + self.gamma1 * self.att(self.ln1(tok), resolution, gate_tok)
            tok = tok + self.gamma2 * self.ffn(self.ln2(tok), resolution)
        return rearrange(tok, "b (h w) c -> b c h w", h=h, w=w)


class _SGRCBiRWKV2D(nn.Module):
    def __init__(
        self,
        n_embd: int,
        hidden_rate: int = 4,
        key_norm: bool = True,
        ffn_first: bool = False,
        shift_pixel: int = 1,
        channel_gamma: float = 1 / 4,
    ) -> None:
        super().__init__()
        self.fwd = _SGRCRWKVBlock2D(n_embd, 2, 0, hidden_rate, key_norm, ffn_first, shift_pixel, channel_gamma)
        self.bwd = _SGRCRWKVBlock2D(n_embd, 2, 1, hidden_rate, key_norm, ffn_first, shift_pixel, channel_gamma)

    def forward(self, x: torch.Tensor, sp_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        a = self.fwd(x, sp_gate)
        x_t = x.transpose(2, 3)
        gate_t = None if sp_gate is None else sp_gate.transpose(2, 3)
        b = self.bwd(x_t, gate_t).transpose(2, 3)
        return a + b


class _SGRCStage(nn.Module):
    def __init__(
        self,
        channels: int,
        spv_name: str,
        n_spixels: int,
        max_spatial: int,
        assignment_topk: int,
        hidden_rate: int,
        key_norm: bool,
        ffn_first: bool,
        shift_pixel: int,
        channel_gamma: float,
        enabled: bool,
        spv_kwargs: Optional[dict[str, object]] = None,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.sp_gate = _SPGateSource(
            channels,
            spv_name,
            n_spixels=n_spixels,
            max_spatial=max_spatial,
            assignment_topk=assignment_topk,
            spv_kwargs=spv_kwargs,
        )
        self.rwkv = (
            _SGRCBiRWKV2D(channels, hidden_rate, key_norm, ffn_first, shift_pixel, channel_gamma)
            if self.enabled
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return self.rwkv(x, self.sp_gate(x))


class HS3R_Net_SGRC_Base(nn.Module):
    def __init__(
        self,
        spv_name: str,
        in_channels: int = 3,
        out_channels: int = 1,
        dims: Sequence[int] = (32, 64, 128, 256, 512),
        rwkv_stages: Sequence[bool] = (False, True, True, True, True),
        hidden_rate: int = 4,
        key_norm: bool = True,
        ffn_first: bool = False,
        shift_pixel: int = 1,
        channel_gamma: float = 1 / 4,
        enable_wkv_cuda: bool = True,
        wkv_t_max: int = T_MAX_DEFAULT,
        wkv_cuda_dir: Optional[str] = None,
        wkv_verbose: bool = False,
        spv_n_spixels: Sequence[int] | int = (64, 64, 64, 64),
        spv_max_spatial: int = 128,
        spv_assignment_topk: Sequence[int] | int = 9,
        spv_use_appearance_energy: bool = True,
        spv_use_spatial_compactness: bool = True,
        spv_use_boundary_confidence: bool = True,
        spv_dense_assignment: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        if spv_name not in SPV_CLASSES:
            raise KeyError(f"Unknown spv_name={spv_name}. Available: {list(SPV_CLASSES)}")
        if enable_wkv_cuda:
            WKV.configure_backend(_WKVBackend(t_max=wkv_t_max, cuda_dir=wkv_cuda_dir, verbose=wkv_verbose))

        c1, c2, c3, c4, c5 = [int(v) for v in dims]
        nsp = _as_tuple4(spv_n_spixels)
        sp_topk = _as_tuple4(spv_assignment_topk, default=9)
        rwkv_stages = tuple(bool(x) for x in rwkv_stages)
        spv_kwargs = {
            "use_appearance_energy": bool(spv_use_appearance_energy),
            "use_spatial_compactness": bool(spv_use_spatial_compactness),
            "use_boundary_confidence": bool(spv_use_boundary_confidence),
            "dense_assignment": bool(spv_dense_assignment),
        } if spv_name == "SPV005" else {}
        self.spv_name = spv_name
        self.inject_mode = "SGRC"

        self.stem = ConvBNAct(in_channels, c1, 3, 1, 1, act="relu")
        self.enc1 = ConvBNAct(c1, c1, 3, 1, 1, act="relu")
        self.enc2 = ConvBNAct(c1, c2, 3, 1, 1, act="relu")
        self.enc3 = ConvBNAct(c2, c3, 3, 1, 1, act="relu")
        self.enc4 = ConvBNAct(c3, c4, 3, 1, 1, act="relu")
        self.enc5 = ConvBNAct(c4, c5, 3, 1, 1, act="relu")
        self.pool = nn.MaxPool2d(2, 2)

        self.rwkv1 = _SGRCStage(c1, spv_name, nsp[0], spv_max_spatial, sp_topk[0], hidden_rate, key_norm, ffn_first, shift_pixel, channel_gamma, rwkv_stages[0], spv_kwargs=spv_kwargs)
        self.rwkv2 = _SGRCStage(c2, spv_name, nsp[1], spv_max_spatial, sp_topk[1], hidden_rate, key_norm, ffn_first, shift_pixel, channel_gamma, rwkv_stages[1], spv_kwargs=spv_kwargs)
        self.rwkv3 = _SGRCStage(c3, spv_name, nsp[2], spv_max_spatial, sp_topk[2], hidden_rate, key_norm, ffn_first, shift_pixel, channel_gamma, rwkv_stages[2], spv_kwargs=spv_kwargs)
        self.rwkv4 = _SGRCStage(c4, spv_name, nsp[3], spv_max_spatial, sp_topk[3], hidden_rate, key_norm, ffn_first, shift_pixel, channel_gamma, rwkv_stages[3], spv_kwargs=spv_kwargs)
        self.rwkv5 = BiRWKV2D(c5, hidden_rate, key_norm, ffn_first, shift_pixel, channel_gamma) if rwkv_stages[4] else nn.Identity()

        self.up5 = UpConv(c5, c4)
        self.fuse5 = FusionConv(c4 * 2, c4)
        self.up4 = UpConv(c4, c3)
        self.fuse4 = FusionConv(c3 * 2, c3)
        self.up3 = UpConv(c3, c2)
        self.fuse3 = FusionConv(c2 * 2, c2)
        self.up2 = UpConv(c2, c1)
        self.fuse2 = FusionConv(c1 * 2, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1, stride=1, padding=0)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.rwkv1(self.enc1(self.stem(x)))
        x2 = self.rwkv2(self.enc2(self.pool(x1)))
        x3 = self.rwkv3(self.enc3(self.pool(x2)))
        x4 = self.rwkv4(self.enc4(self.pool(x3)))
        x5 = self.rwkv5(self.enc5(self.pool(x4)))
        return x1, x2, x3, x4, x5

    def decode(self, feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2, x3, x4, x5 = feats
        d5 = self.fuse5(torch.cat([x4, self.up5(x5)], dim=1))
        d4 = self.fuse4(torch.cat([x3, self.up4(d5)], dim=1))
        d3 = self.fuse3(torch.cat([x2, self.up3(d4)], dim=1))
        d2 = self.fuse2(torch.cat([x1, self.up2(d3)], dim=1))
        return self.head(d2)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encode(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
