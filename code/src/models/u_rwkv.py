from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from .registry import register_model

T_MAX_DEFAULT = 16384


# ============================================================
# Basic blocks
# ============================================================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1,
                 act: str = "relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class UpConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class FusionConv(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ============================================================
# Shift
# ============================================================

def q_shift(x: torch.Tensor, shift_pixel: int = 1, gamma: float = 1 / 4,
            resolution: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    assert gamma <= 1 / 4
    B, N, C = x.shape
    if resolution is None:
        s = int(N ** 0.5)
        if s * s != N:
            raise ValueError(f"q_shift: cannot infer square resolution from N={N}")
        resolution = (s, s)
    H, W = resolution

    x2 = x.transpose(1, 2).reshape(B, C, H, W)
    out = torch.zeros_like(x2)

    c1 = int(C * gamma)
    c2 = int(C * gamma * 2)
    c3 = int(C * gamma * 3)
    c4 = int(C * gamma * 4)

    out[:, 0:c1, :, shift_pixel:W] = x2[:, 0:c1, :, 0:W - shift_pixel]
    out[:, c1:c2, :, 0:W - shift_pixel] = x2[:, c1:c2, :, shift_pixel:W]
    out[:, c2:c3, shift_pixel:H, :] = x2[:, c2:c3, 0:H - shift_pixel, :]
    out[:, c3:c4, 0:H - shift_pixel, :] = x2[:, c3:c4, shift_pixel:H, :]
    out[:, c4:, ...] = x2[:, c4:, ...]

    return out.flatten(2).transpose(1, 2)


# ============================================================
# CUDA extension backend
# ============================================================

class _WKVBackend:
    def __init__(self, t_max: int = T_MAX_DEFAULT, cuda_dir: Optional[str] = None, verbose: bool = False):
        self.t_max = int(t_max)
        self.cuda_dir = cuda_dir
        self.verbose = verbose
        self.mod = None

    def _try_load(self):
        if self.mod is not None:
            return

        # # 1) prebuilt
        # try:
        #     import wkv as wkv_mod  # type: ignore
        #     self.mod = wkv_mod
        #     return
        # except Exception:
        #     pass

        # 2) build from src/models/wkv_op.cpp + wkv_cuda.cu
        cuda_dir = self.cuda_dir
        if cuda_dir is None:
            cuda_dir = str(Path(__file__).resolve().parent)

        cpp = Path(cuda_dir) / "wkv_op.cpp"
        cu = Path(cuda_dir) / "wkv_cuda.cu"
        if not (cpp.exists() and cu.exists()):
            raise RuntimeError(
                "U-RWKV needs CUDA WKV extension but sources not found.\n"
                f"Expected:\n  {cpp}\n  {cu}\n"
                "Fix: put them into src/models/ or set wkv_cuda_dir."
            )

        from torch.utils.cpp_extension import load

        extra_cuda_cflags = [
            "--use_fast_math",
            "--maxrregcount=60",
            "-O3",
            "-Xptxas=-O3",
            f"-DTmax={self.t_max}",
        ]

        if not torch.cuda.is_available():
            raise RuntimeError("WKV CUDA extension requested, but CUDA is not available.")
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        build_tag = re.sub(r"[^0-9A-Za-z_]", "_", os.environ.get("WKV_BUILD_TAG", "")).strip("_")
        extension_name = f"wkv_tmax{self.t_max}_sm{major}{minor}"
        if build_tag:
            extension_name = f"{extension_name}_{build_tag}"

        self.mod = load(
            name=extension_name,
            sources=[str(cpp), str(cu)],
            verbose=self.verbose,
            extra_cflags=["-O3"],
            extra_cuda_cflags=extra_cuda_cflags,
        )

    def forward(self, B: int, T: int, C: int,
                w: torch.Tensor, u: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self._try_load()
        assert self.mod is not None
        y = torch.empty((B, T, C), device=k.device, dtype=torch.float32).contiguous()
        self.mod.forward(B, T, C, w, u, k, v, y)  # type: ignore
        return y

    def backward(self, B: int, T: int, C: int,
                 w: torch.Tensor, u: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 gy: torch.Tensor):
        self._try_load()
        assert self.mod is not None
        gw = torch.zeros((B, C), device=gy.device, dtype=torch.float32).contiguous()
        gu = torch.zeros((B, C), device=gy.device, dtype=torch.float32).contiguous()
        gk = torch.zeros((B, T, C), device=gy.device, dtype=torch.float32).contiguous()
        gv = torch.zeros((B, T, C), device=gy.device, dtype=torch.float32).contiguous()
        self.mod.backward(B, T, C, w, u, k, v, gy, gw, gu, gk, gv)  # type: ignore
        return gw, gu, gk, gv


class WKV(torch.autograd.Function):
    backend: Optional[_WKVBackend] = None

    @staticmethod
    def configure_backend(backend: _WKVBackend):
        WKV.backend = backend

    @staticmethod
    def forward(ctx, B: int, T: int, C: int,
                w: torch.Tensor, u: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if WKV.backend is None:
            raise RuntimeError("WKV backend not configured.")
        if T > WKV.backend.t_max:
            raise RuntimeError(f"WKV T={T} exceeds Tmax={WKV.backend.t_max}. Increase wkv_t_max or reduce tokens.")

        half_mode = (w.dtype == torch.float16)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.B, ctx.T, ctx.C = B, T, C
        ctx.half_mode, ctx.bf_mode = half_mode, bf_mode

        w0 = w.float().contiguous()
        u0 = u.float().contiguous()
        k0 = k.float().contiguous()
        v0 = v.float().contiguous()
        ctx.save_for_backward(w0, u0, k0, v0)

        y = WKV.backend.forward(B, T, C, w0, u0, k0, v0)

        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy: torch.Tensor):
        if WKV.backend is None:
            raise RuntimeError("WKV backend not configured.")
        B, T, C = ctx.B, ctx.T, ctx.C
        w0, u0, k0, v0 = ctx.saved_tensors
        gy0 = gy.float().contiguous()

        gw, gu, gk, gv = WKV.backend.backward(B, T, C, w0, u0, k0, v0, gy0)

        # sum across batch
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)

        if ctx.half_mode:
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        if ctx.bf_mode:
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        return (None, None, None, gw, gu, gk, gv)


def RUN_WKV(B: int, T: int, C: int,
            w: torch.Tensor, u: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if WKV.backend is None:
        # Portable fallback for CPU smoke tests or environments where the CUDA
        # extension has not been compiled yet. Training configs should keep the
        # CUDA backend enabled for speed.
        y = torch.empty_like(v)
        aa = torch.full((B, C), -1e30, device=v.device, dtype=torch.float32)
        bb = torch.zeros((B, C), device=v.device, dtype=torch.float32)
        pp = torch.full((B, C), -1e30, device=v.device, dtype=torch.float32)
        ww = -torch.exp(w.float()).view(1, C)
        uu = u.float().view(1, C)
        kk = k.float()
        vv = v.float()
        for t in range(T):
            kt = kk[:, t, :]
            vt = vv[:, t, :]
            p = torch.maximum(pp, uu + kt)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(uu + kt - p)
            y[:, t, :] = ((e1 * aa + e2 * vt) / (e1 * bb + e2)).to(v.dtype)

            p = torch.maximum(pp + ww, kt)
            e1 = torch.exp(pp + ww - p)
            e2 = torch.exp(kt - p)
            aa = e1 * aa + e2 * vt
            bb = e1 * bb + e2
            pp = p
        return y
    return WKV.apply(B, T, C, w, u, k, v)


# ============================================================
# RWKV blocks
# ============================================================

class SpatialInteractionMix(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, layer_id: int,
                 shift_pixel: int = 1, channel_gamma: float = 1/4,
                 key_norm: bool = True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        attn_sz = n_embd

        with torch.no_grad():
            ratio_0_to_1 = (self.layer_id / (self.n_layer - 1 + 1e-6))
            ratio_1_to_almost0 = (1.0 - (self.layer_id / (self.n_layer + 1e-6)))

            decay_speed = torch.ones(self.n_embd)
            for h in range(self.n_embd):
                decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1 + 1e-6)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.spatial_decay = nn.Parameter(decay_speed)

            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)], dtype=torch.float32) * 0.5)
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

    def forward(self, x: torch.Tensor, resolution: Tuple[int, int]) -> torch.Tensor:
        B, T, C = x.shape
        if self.shift_pixel > 0:
            xx = q_shift(x, self.shift_pixel, self.channel_gamma, resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk, xv, xr = x, x, x

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        w = (self.spatial_decay / T).to(x.device)
        u = (self.spatial_first / T).to(x.device)

        rwkv = RUN_WKV(B, T, C, w, u, k, v)
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        rwkv = sr * rwkv
        return self.output(rwkv)


class SpectralMixer(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, layer_id: int,
                 shift_pixel: int = 1, channel_gamma: float = 1/4,
                 hidden_rate: int = 4, key_norm: bool = True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd

        with torch.no_grad():
            ratio_1_to_almost0 = (1.0 - (self.layer_id / (self.n_layer + 1e-6)))
            x = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                x[0, 0, i] = i / self.n_embd
            self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        self.shift_pixel = int(shift_pixel)
        self.channel_gamma = float(channel_gamma)

        hidden_sz = int(hidden_rate) * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        self.key_norm = nn.LayerNorm(hidden_sz) if key_norm else None

    def forward(self, x: torch.Tensor, resolution: Tuple[int, int]) -> torch.Tensor:
        if self.shift_pixel > 0:
            xx = q_shift(x, self.shift_pixel, self.channel_gamma, resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk, xr = x, x

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class RWKVBlock2D(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, layer_id: int,
                 hidden_rate: int = 4, key_norm: bool = False,
                 ffn_first: bool = False,
                 shift_pixel: int = 1, channel_gamma: float = 1/4):
        super().__init__()
        self.ffn_first = ffn_first
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.att = SpatialInteractionMix(
            n_embd=n_embd, n_layer=n_layer, layer_id=layer_id,
            shift_pixel=shift_pixel, channel_gamma=channel_gamma,
            key_norm=key_norm,
        )
        self.ffn = SpectralMixer(
            n_embd=n_embd, n_layer=n_layer, layer_id=layer_id,
            shift_pixel=shift_pixel, channel_gamma=channel_gamma,
            hidden_rate=hidden_rate, key_norm=key_norm,
        )
        self.gamma1 = nn.Parameter(torch.ones((n_embd,)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd,)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tok = rearrange(x, "b c h w -> b (h w) c")
        resolution = (h, w)

        if self.ffn_first:
            tok = tok + self.gamma2 * self.ffn(self.ln2(tok), resolution)
            tok = tok + self.gamma1 * self.att(self.ln1(tok), resolution)
        else:
            tok = tok + self.gamma1 * self.att(self.ln1(tok), resolution)
            tok = tok + self.gamma2 * self.ffn(self.ln2(tok), resolution)

        return rearrange(tok, "b (h w) c -> b c h w", h=h, w=w)


class BiRWKV2D(nn.Module):
    def __init__(self, n_embd: int, hidden_rate: int = 4,
                 key_norm: bool = True, ffn_first: bool = False,
                 shift_pixel: int = 1, channel_gamma: float = 1/4):
        super().__init__()
        self.fwd = RWKVBlock2D(n_embd=n_embd, n_layer=2, layer_id=0,
                              hidden_rate=hidden_rate, key_norm=key_norm,
                              ffn_first=ffn_first, shift_pixel=shift_pixel, channel_gamma=channel_gamma)
        self.bwd = RWKVBlock2D(n_embd=n_embd, n_layer=2, layer_id=1,
                              hidden_rate=hidden_rate, key_norm=key_norm,
                              ffn_first=ffn_first, shift_pixel=shift_pixel, channel_gamma=channel_gamma)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        a = self.fwd(z)
        z_t = z.transpose(2, 3)
        b = self.bwd(z_t).transpose(2, 3)
        return a + b


# ============================================================
# U-RWKV UNet
# ============================================================

class URWKV_UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        dims=(32, 64, 128, 256, 512),
        rwkv_stages=(False, True, True, True, True),
        hidden_rate: int = 4,
        key_norm: bool = True,
        ffn_first: bool = False,
        shift_pixel: int = 1,
        channel_gamma: float = 1/4,
        enable_wkv_cuda: bool = True,
        wkv_t_max: int = T_MAX_DEFAULT,
        wkv_cuda_dir: Optional[str] = None,
        wkv_verbose: bool = False,
    ):
        super().__init__()

        if enable_wkv_cuda:
            backend = _WKVBackend(t_max=wkv_t_max, cuda_dir=wkv_cuda_dir, verbose=wkv_verbose)
            WKV.configure_backend(backend)

        c1, c2, c3, c4, c5 = dims

        self.stem = ConvBNAct(in_channels, c1, 3, 1, 1, act="relu")
        self.enc1 = ConvBNAct(c1, c1, 3, 1, 1, act="relu")

        self.enc2 = ConvBNAct(c1, c2, 3, 1, 1, act="relu")
        self.enc3 = ConvBNAct(c2, c3, 3, 1, 1, act="relu")
        self.enc4 = ConvBNAct(c3, c4, 3, 1, 1, act="relu")
        self.enc5 = ConvBNAct(c4, c5, 3, 1, 1, act="relu")

        self.rwkv_stages = tuple(bool(x) for x in rwkv_stages)
        self.rwkv1 = BiRWKV2D(c1, hidden_rate, key_norm, ffn_first, shift_pixel, channel_gamma) if self.rwkv_stages[0] else nn.Identity()
        self.rwkv2 = BiRWKV2D(c2, hidden_rate, key_norm, ffn_first, shift_pixel, channel_gamma) if self.rwkv_stages[1] else nn.Identity()
        self.rwkv3 = BiRWKV2D(c3, hidden_rate, key_norm, ffn_first, shift_pixel, channel_gamma) if self.rwkv_stages[2] else nn.Identity()
        self.rwkv4 = BiRWKV2D(c4, hidden_rate, key_norm, ffn_first, shift_pixel, channel_gamma) if self.rwkv_stages[3] else nn.Identity()
        self.rwkv5 = BiRWKV2D(c5, hidden_rate, key_norm, ffn_first, shift_pixel, channel_gamma) if self.rwkv_stages[4] else nn.Identity()

        self.pool = nn.MaxPool2d(2, 2)

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
        x1 = self.enc1(self.stem(x))
        x1 = self.rwkv1(x1)

        x2 = self.enc2(self.pool(x1))
        x2 = self.rwkv2(x2)

        x3 = self.enc3(self.pool(x2))
        x3 = self.rwkv3(x3)

        x4 = self.enc4(self.pool(x3))
        x4 = self.rwkv4(x4)

        x5 = self.enc5(self.pool(x4))
        x5 = self.rwkv5(x5)

        return x1, x2, x3, x4, x5

    def decode(self, feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2, x3, x4, x5 = feats

        d5 = self.up5(x5)
        d5 = self.fuse5(torch.cat([x4, d5], dim=1))

        d4 = self.up4(d5)
        d4 = self.fuse4(torch.cat([x3, d4], dim=1))

        d3 = self.up3(d4)
        d3 = self.fuse3(torch.cat([x2, d3], dim=1))

        d2 = self.up2(d3)
        d2 = self.fuse2(torch.cat([x1, d2], dim=1))

        return self.head(d2)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encode(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


@register_model("u_rwkv_unet")
@register_model("U_RWKV")
def build_u_rwkv_unet(
    in_channels: int = 3,
    out_channels: int = 1,
    dims=(32, 64, 128, 256, 512),
    rwkv_stages=(False, True, True, True, True),
    hidden_rate: int = 4,
    key_norm: bool = True,
    ffn_first: bool = False,
    shift_pixel: int = 1,
    channel_gamma: float = 1/4,
    enable_wkv_cuda: bool = True,
    wkv_t_max: int = T_MAX_DEFAULT,
    wkv_cuda_dir: Optional[str] = None,
    wkv_verbose: bool = False,
):
    return URWKV_UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        dims=dims,
        rwkv_stages=rwkv_stages,
        hidden_rate=hidden_rate,
        key_norm=key_norm,
        ffn_first=ffn_first,
        shift_pixel=shift_pixel,
        channel_gamma=channel_gamma,
        enable_wkv_cuda=enable_wkv_cuda,
        wkv_t_max=wkv_t_max,
        wkv_cuda_dir=wkv_cuda_dir,
        wkv_verbose=wkv_verbose,
    )
