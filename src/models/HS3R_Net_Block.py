from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.models.S3HNet.diff_slic import DiffSLIC, spixel_upsampling
from src.models.u_rwkv import RUN_WKV, T_MAX_DEFAULT, WKV, _WKVBackend, q_shift


# Common utility blocks.
class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# Feature interaction modules. These are not superpixel versions.
class TopKSparseAttention(nn.Module):
    """The original TKSA-style channel attention, kept as a baseline block."""

    def __init__(self, dim: int, num_heads: int = 4, bias: bool = False) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = int(num_heads)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.0)
        self.attn1 = nn.Parameter(torch.tensor([0.2]))
        self.attn2 = nn.Parameter(torch.tensor([0.2]))
        self.attn3 = nn.Parameter(torch.tensor([0.3]))
        self.attn4 = nn.Parameter(torch.tensor([0.3]))

    @staticmethod
    def _topk_attention(attn: torch.Tensor, keep: int) -> torch.Tensor:
        keep = max(1, min(int(keep), attn.shape[-1]))
        index = torch.topk(attn, k=keep, dim=-1, largest=True)[1]
        mask = torch.zeros_like(attn, dtype=torch.bool)
        mask.scatter_(-1, index, True)
        attn = torch.where(mask, attn, torch.full_like(attn, float("-inf")))
        return attn.softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        channels_per_head = q.shape[-2]
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn1 = self._topk_attention(attn, channels_per_head / 2)
        attn2 = self._topk_attention(attn, channels_per_head * 2 / 3)
        attn3 = self._topk_attention(attn, channels_per_head * 3 / 4)
        attn4 = self._topk_attention(attn, channels_per_head * 4 / 5)

        out = (
            (attn1 @ v) * self.attn1
            + (attn2 @ v) * self.attn2
            + (attn3 @ v) * self.attn3
            + (attn4 @ v) * self.attn4
        )
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        return self.project_out(self.attn_drop(out))


class Top_k_rwkv(nn.Module):
    """Top-k sparse gating around EWKV/RWKV token mixing.

    This block keeps the same spatial tensor API as TKSA, but replaces the
    quadratic QK attention matrix with WKV sequence mixing. The top-k part is
    applied as sparse channel selection over the EWKV response, so it stays
    lightweight and can be dropped into 2D feature blocks.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        bias: bool = False,
        topk_ratios: Tuple[float, float, float, float] = (0.5, 2 / 3, 3 / 4, 4 / 5),
        shift_pixel: int = 1,
        channel_gamma: float = 0.25,
        key_norm: bool = True,
        enable_wkv_cuda: bool = False,
        wkv_t_max: int = T_MAX_DEFAULT,
        wkv_cuda_dir: Optional[str] = None,
        wkv_verbose: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.topk_ratios = tuple(float(r) for r in topk_ratios)
        self.shift_pixel = int(shift_pixel)
        self.channel_gamma = float(channel_gamma)

        if enable_wkv_cuda:
            WKV.configure_backend(_WKVBackend(wkv_t_max, cuda_dir=wkv_cuda_dir, verbose=wkv_verbose))

        self.local = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim) if key_norm else nn.Identity()

        self.spatial_decay = nn.Parameter(torch.linspace(-5.0, 3.0, dim))
        self.spatial_first = nn.Parameter(torch.zeros(dim))
        self.mix_k = nn.Parameter(torch.linspace(0.0, 1.0, dim).view(1, 1, dim))
        self.mix_v = nn.Parameter(torch.linspace(0.0, 1.0, dim).view(1, 1, dim))
        self.mix_r = nn.Parameter(torch.linspace(0.0, 1.0, dim).view(1, 1, dim))

        self.branch_weights = nn.Parameter(torch.tensor([0.2, 0.2, 0.3, 0.3], dtype=torch.float32))
        self.project_out = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    @staticmethod
    def _channel_topk(x: torch.Tensor, ratio: float) -> torch.Tensor:
        b, c, h, w = x.shape
        keep = max(1, min(int(c * ratio), c))
        score = x.abs().mean(dim=(2, 3), keepdim=True)
        index = torch.topk(score.flatten(1), k=keep, dim=1, largest=True)[1]
        mask = torch.zeros(b, c, device=x.device, dtype=x.dtype)
        mask.scatter_(1, index, 1.0)
        return x * mask.view(b, c, 1, 1)

    def _ewkv(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        token = rearrange(self.local(x), "b c h w -> b (h w) c")
        if self.shift_pixel > 0:
            shifted = q_shift(token, self.shift_pixel, self.channel_gamma, (h, w))
            xk = token * self.mix_k + shifted * (1.0 - self.mix_k)
            xv = token * self.mix_v + shifted * (1.0 - self.mix_v)
            xr = token * self.mix_r + shifted * (1.0 - self.mix_r)
        else:
            xk, xv, xr = token, token, token

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))
        w_decay = (self.spatial_decay / max(token.shape[1], 1)).to(device=x.device, dtype=x.dtype)
        u_first = (self.spatial_first / max(token.shape[1], 1)).to(device=x.device, dtype=x.dtype)
        mixed = RUN_WKV(b, token.shape[1], c, w_decay, u_first, k, v)
        mixed = self.output(r * self.norm(mixed))
        return rearrange(mixed, "b (h w) c -> b c h w", h=h, w=w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ewkv = self._ewkv(x)
        weights = torch.softmax(self.branch_weights, dim=0)
        sparse = 0.0
        for weight, ratio in zip(weights, self.topk_ratios):
            sparse = sparse + weight * self._channel_topk(ewkv, ratio)
        return self.project_out(sparse) + x


@dataclass
class SuperpixelOutput:
    vertices: torch.Tensor
    assignment: torch.Tensor
    pixel_features: torch.Tensor
    density: Optional[torch.Tensor] = None
    center_indices: Optional[torch.Tensor] = None


def _sp_grid_size(n_spixels: int, height: int, width: int) -> tuple[int, int]:
    height_s = int((n_spixels * height / max(width, 1)) ** 0.5)
    width_s = int((n_spixels * width / max(height, 1)) ** 0.5)
    return max(1, height_s), max(1, width_s)


def _coord_grid(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    batch: int = 1,
) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coord = torch.stack([yy, xx], dim=0).unsqueeze(0)
    return coord.expand(batch, -1, -1, -1)


def _topk_reconstruct(
    centers: torch.Tensor,
    logits: torch.Tensor,
    height: int,
    width: int,
    topk: int = 9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, c, _, _ = centers.shape
    n_tokens = logits.shape[1]
    topk = max(1, min(int(topk), logits.shape[-1]))
    values, indices = torch.topk(logits, k=topk, dim=-1, largest=True)
    weights = torch.softmax(values, dim=-1)

    centers_flat = centers.flatten(2).transpose(1, 2)
    gathered = torch.gather(
        centers_flat.unsqueeze(1).expand(-1, n_tokens, -1, -1),
        dim=2,
        index=indices.unsqueeze(-1).expand(-1, -1, -1, c),
    )
    pixels = (gathered * weights.unsqueeze(-1)).sum(dim=2)
    pixel_features = pixels.transpose(1, 2).reshape(b, c, height, width)
    assignment = weights.transpose(1, 2).reshape(b, topk, height, width)
    center_indices = indices.transpose(1, 2).reshape(b, topk, height, width)
    return pixel_features, assignment, center_indices


def _dense_reconstruct(
    centers: torch.Tensor,
    logits: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, c, _, _ = centers.shape
    n_tokens, n_centers = logits.shape[1], logits.shape[2]
    weights = torch.softmax(logits, dim=-1)
    centers_flat = centers.flatten(2).transpose(1, 2)
    pixels = torch.bmm(weights, centers_flat)
    pixel_features = pixels.transpose(1, 2).reshape(b, c, height, width)
    assignment = weights.transpose(1, 2).reshape(b, n_centers, height, width)
    center_indices = (
        torch.arange(n_centers, device=logits.device)
        .view(1, n_centers, 1, 1)
        .expand(b, -1, height, width)
    )
    return pixel_features, assignment, center_indices


class _GlobalSoftSuperpixelBase(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_channels: int,
        n_spixels: int,
        assignment_topk: int,
        dense_assignment: bool = False,
    ) -> None:
        super().__init__()
        self.n_spixels = int(n_spixels)
        self.assignment_topk = int(assignment_topk)
        self.dense_assignment = bool(dense_assignment)
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, embed_channels, kernel_size=3, padding=1),
            ConvBNAct(embed_channels, embed_channels, kernel_size=3, padding=1),
        )

    def _centers_from_pool(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = feat.shape
        hs, ws = _sp_grid_size(self.n_spixels, h, w)
        centers = F.adaptive_avg_pool2d(feat, output_size=(hs, ws))
        pix_coord = _coord_grid(h, w, feat.device, feat.dtype, b).flatten(2).transpose(1, 2)
        ctr_coord = _coord_grid(hs, ws, feat.device, feat.dtype, b).flatten(2).transpose(1, 2)
        return centers, pix_coord, ctr_coord

    def _emit(
        self,
        centers: torch.Tensor,
        logits: torch.Tensor,
        height: int,
        width: int,
        density: Optional[torch.Tensor] = None,
    ) -> SuperpixelOutput:
        if self.dense_assignment:
            pixel_features, assignment, center_indices = _dense_reconstruct(centers, logits, height, width)
        else:
            pixel_features, assignment, center_indices = _topk_reconstruct(
                centers, logits, height, width, topk=self.assignment_topk
            )
        return SuperpixelOutput(
            vertices=centers,
            assignment=assignment,
            pixel_features=pixel_features,
            density=density,
            center_indices=center_indices,
        )


# Superpixel algorithm versions. SPVxxx names are reserved for this family.
class SPV001(nn.Module):
    """Baseline wrapper of the current DiffSLIC superpixel operator."""

    def __init__(
        self,
        in_channels: int,
        embed_channels: int = 32,
        n_spixels: int = 256,
        slic_iters: int = 5,
        slic_dist: str = "slic",
        compactness: float = 0.17,
        lambda_edge: float = 0.5,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            ConvBNAct(in_channels, embed_channels, kernel_size=3, padding=1),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=1, bias=True),
        )
        self.edge = nn.Conv2d(embed_channels, 1, kernel_size=3, padding=1, bias=True)
        self.slic = DiffSLIC(
            n_spixels=n_spixels,
            n_iter=slic_iters,
            sim_type=slic_dist,
            compactness=compactness,
            lambda_edge=lambda_edge,
        )

    def forward(self, x: torch.Tensor) -> SuperpixelOutput:
        feat = self.proj(x)
        edge_logits = self.edge(feat)
        vertices, assignment, _ = self.slic(feat, edge_logits=edge_logits)
        pixel_features = spixel_upsampling(vertices, assignment)
        return SuperpixelOutput(vertices=vertices, assignment=assignment, pixel_features=pixel_features)


class SPV002(nn.Module):
    """Task-aware DiffSLIC wrapper with EWKV-enhanced features before clustering."""

    def __init__(
        self,
        in_channels: int,
        embed_channels: int = 32,
        n_spixels: int = 256,
        slic_iters: int = 5,
        compactness: float = 0.17,
        lambda_edge: float = 0.8,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.stem = ConvBNAct(in_channels, embed_channels, kernel_size=3, padding=1)
        self.ewkv_refine = Top_k_rwkv(embed_channels, num_heads=num_heads, enable_wkv_cuda=False)
        self.edge = nn.Sequential(
            ConvBNAct(embed_channels, embed_channels, kernel_size=3, padding=1, groups=embed_channels),
            nn.Conv2d(embed_channels, 1, kernel_size=1, bias=True),
        )
        self.slic = DiffSLIC(
            n_spixels=n_spixels,
            n_iter=slic_iters,
            sim_type="slic",
            compactness=compactness,
            lambda_edge=lambda_edge,
        )

    def forward(self, x: torch.Tensor) -> SuperpixelOutput:
        feat = self.ewkv_refine(self.stem(x))
        edge_logits = self.edge(feat)
        vertices, assignment, _ = self.slic(feat, edge_logits=edge_logits)
        pixel_features = spixel_upsampling(vertices, assignment)
        return SuperpixelOutput(vertices=vertices, assignment=assignment, pixel_features=pixel_features)


class SPV003(nn.Module):
    """Adaptive semantic DiffSLIC with learnable seed-density initialization.

    SPV003 keeps the DiffSLIC soft assignment step, but replaces uniform center
    initialization with density-weighted centers. In practice this lets the
    network place stronger initial superpixel evidence around task-relevant
    structure while preserving differentiability.
    """

    def __init__(
        self,
        in_channels: int,
        embed_channels: int = 32,
        n_spixels: int = 256,
        slic_iters: int = 5,
        compactness: float = 0.17,
        lambda_edge: float = 0.8,
        num_heads: int = 4,
        density_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_spixels = int(n_spixels)
        self.density_temperature = float(density_temperature)
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, embed_channels, kernel_size=3, padding=1),
            ConvBNAct(embed_channels, embed_channels, kernel_size=3, padding=1),
        )
        self.semantic_refine = Top_k_rwkv(embed_channels, num_heads=num_heads, enable_wkv_cuda=False)
        self.density_head = nn.Sequential(
            ConvBNAct(embed_channels, embed_channels, kernel_size=3, padding=1, groups=embed_channels),
            nn.Conv2d(embed_channels, 1, kernel_size=1, bias=True),
        )
        self.edge_head = nn.Sequential(
            ConvBNAct(embed_channels, embed_channels, kernel_size=3, padding=1, groups=embed_channels),
            nn.Conv2d(embed_channels, 1, kernel_size=1, bias=True),
        )
        self.center_proj = ConvBNAct(embed_channels, embed_channels, kernel_size=1, padding=0)
        self.slic = DiffSLIC(
            n_spixels=n_spixels,
            n_iter=slic_iters,
            sim_type="slic",
            compactness=compactness,
            lambda_edge=lambda_edge,
        )

    def _grid_size(self, height: int, width: int) -> tuple[int, int]:
        height_s = int((self.n_spixels * height / max(width, 1)) ** 0.5)
        width_s = int((self.n_spixels * width / max(height, 1)) ** 0.5)
        return max(1, height_s), max(1, width_s)

    def _init_centers(self, feat: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        _, _, height, width = feat.shape
        height_s, width_s = self._grid_size(height, width)
        weight = density.clamp_min(1e-4)
        numerator = F.adaptive_avg_pool2d(feat * weight, output_size=(height_s, width_s))
        denominator = F.adaptive_avg_pool2d(weight, output_size=(height_s, width_s)).clamp_min(1e-4)
        return self.center_proj(numerator / denominator)

    def forward(self, x: torch.Tensor) -> SuperpixelOutput:
        feat = self.semantic_refine(self.stem(x))
        density_logits = self.density_head(feat)
        density = torch.sigmoid(density_logits / max(self.density_temperature, 1e-4))
        edge_logits = self.edge_head(feat)
        centers = self._init_centers(feat, density)
        vertices, assignment, _ = self.slic(feat, clst_feats=centers, edge_logits=edge_logits)
        pixel_features = spixel_upsampling(vertices, assignment)
        return SuperpixelOutput(
            vertices=vertices,
            assignment=assignment,
            pixel_features=pixel_features,
            density=density,
        )


class SPV004(_GlobalSoftSuperpixelBase):
    """SSN-style differentiable superpixel sampling.

    Inspired by Superpixel Sampling Networks, this version learns an embedding
    and performs soft assignment from pixels to learned grid prototypes.
    """

    def __init__(
        self,
        in_channels: int,
        embed_channels: int = 32,
        n_spixels: int = 256,
        assignment_topk: int = 9,
        compactness: float = 0.12,
        temperature: float = 0.07,
        dense_assignment: bool = False,
    ) -> None:
        super().__init__(in_channels, embed_channels, n_spixels, assignment_topk, dense_assignment=dense_assignment)
        self.embedding = ConvBNAct(embed_channels, embed_channels, kernel_size=1, padding=0)
        self.compactness = float(compactness)
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor) -> SuperpixelOutput:
        feat = self.embedding(self.stem(x))
        b, c, h, w = feat.shape
        centers, pix_coord, ctr_coord = self._centers_from_pool(feat)

        pix = F.normalize(feat.flatten(2).transpose(1, 2), dim=-1)
        ctr = F.normalize(centers.flatten(2).transpose(1, 2), dim=-1)
        sim = torch.bmm(pix, ctr.transpose(1, 2)) / max(self.temperature, 1e-4)
        spatial = torch.cdist(pix_coord, ctr_coord, p=2).pow(2)
        logits = sim - self.compactness * spatial
        return self._emit(centers, logits, h, w)


class SPV005(_GlobalSoftSuperpixelBase):
    """Differentiable SEEDS-style energy sampling.

    SEEDS optimizes region histogram/appearance energy. This soft variant uses
    learned feature energy plus boundary confidence, then samples sparse soft
    assignments to region prototypes.
    """

    def __init__(
        self,
        in_channels: int,
        embed_channels: int = 32,
        n_spixels: int = 256,
        assignment_topk: int = 9,
        compactness: float = 0.10,
        lambda_edge: float = 0.7,
        temperature: float = 0.08,
        use_appearance_energy: bool = True,
        use_spatial_compactness: bool = True,
        use_boundary_confidence: bool = True,
        dense_assignment: bool = False,
    ) -> None:
        super().__init__(in_channels, embed_channels, n_spixels, assignment_topk, dense_assignment=dense_assignment)
        self.energy_proj = nn.Sequential(
            ConvBNAct(embed_channels, embed_channels, kernel_size=3, padding=1, groups=embed_channels),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.GELU(),
        )
        self.edge_head = nn.Conv2d(embed_channels, 1, kernel_size=3, padding=1, bias=True)
        self.compactness = float(compactness)
        self.lambda_edge = float(lambda_edge)
        self.temperature = float(temperature)
        self.use_appearance_energy = bool(use_appearance_energy)
        self.use_spatial_compactness = bool(use_spatial_compactness)
        self.use_boundary_confidence = bool(use_boundary_confidence)

    def forward(self, x: torch.Tensor) -> SuperpixelOutput:
        feat = self.energy_proj(self.stem(x))
        b, _, h, w = feat.shape
        edge = torch.sigmoid(self.edge_head(feat))
        centers, pix_coord, ctr_coord = self._centers_from_pool(feat)
        edge_centers = F.adaptive_avg_pool2d(edge, output_size=centers.shape[-2:]).flatten(2)

        pix = F.normalize(feat.flatten(2).transpose(1, 2), dim=-1)
        ctr = F.normalize(centers.flatten(2).transpose(1, 2), dim=-1)
        appearance_energy = torch.bmm(pix, ctr.transpose(1, 2)) / max(self.temperature, 1e-4)
        spatial_energy = torch.cdist(pix_coord, ctr_coord, p=2).pow(2)
        edge_pixels = edge.flatten(2).transpose(1, 2)
        edge_energy = edge_pixels + edge_centers.squeeze(1).unsqueeze(1)
        logits = torch.zeros_like(appearance_energy)
        if self.use_appearance_energy:
            logits = logits + appearance_energy
        if self.use_spatial_compactness:
            logits = logits - self.compactness * spatial_energy
        if self.use_boundary_confidence:
            logits = logits - self.lambda_edge * edge_energy
        return self._emit(centers, logits, h, w)


class SPV006(_GlobalSoftSuperpixelBase):
    """Differentiable SNIC-style non-iterative clustering.

    SNIC removes the iterative update loop. This version follows that spirit:
    one learned seed grid, one soft feature-spatial distance assignment.
    """

    def __init__(
        self,
        in_channels: int,
        embed_channels: int = 32,
        n_spixels: int = 256,
        assignment_topk: int = 9,
        compactness: float = 0.25,
        temperature: float = 0.10,
        dense_assignment: bool = False,
    ) -> None:
        super().__init__(in_channels, embed_channels, n_spixels, assignment_topk, dense_assignment=dense_assignment)
        self.seed_proj = ConvBNAct(embed_channels, embed_channels, kernel_size=1, padding=0)
        self.compactness = float(compactness)
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor) -> SuperpixelOutput:
        feat = self.seed_proj(self.stem(x))
        b, _, h, w = feat.shape
        centers, pix_coord, ctr_coord = self._centers_from_pool(feat)

        pix = feat.flatten(2).transpose(1, 2)
        ctr = centers.flatten(2).transpose(1, 2)
        feature_dist = torch.cdist(pix, ctr, p=2).pow(2)
        spatial_dist = torch.cdist(pix_coord, ctr_coord, p=2).pow(2)
        logits = -(feature_dist + self.compactness * spatial_dist) / max(self.temperature, 1e-4)
        return self._emit(centers, logits, h, w)


class SPV007(_GlobalSoftSuperpixelBase):
    """Soft watershed / rooted-spanning-superpixel style assignment.

    This variant learns root confidence and boundary cost, then assigns pixels
    to roots using a differentiable geodesic-like soft cost.
    """

    def __init__(
        self,
        in_channels: int,
        embed_channels: int = 32,
        n_spixels: int = 256,
        assignment_topk: int = 9,
        compactness: float = 0.16,
        lambda_edge: float = 1.0,
        lambda_root: float = 0.5,
        temperature: float = 0.08,
        dense_assignment: bool = False,
    ) -> None:
        super().__init__(in_channels, embed_channels, n_spixels, assignment_topk, dense_assignment=dense_assignment)
        self.refine = Top_k_rwkv(embed_channels, num_heads=4, enable_wkv_cuda=False)
        self.edge_head = nn.Conv2d(embed_channels, 1, kernel_size=3, padding=1, bias=True)
        self.root_head = nn.Conv2d(embed_channels, 1, kernel_size=3, padding=1, bias=True)
        self.compactness = float(compactness)
        self.lambda_edge = float(lambda_edge)
        self.lambda_root = float(lambda_root)
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor) -> SuperpixelOutput:
        feat = self.refine(self.stem(x))
        b, _, h, w = feat.shape
        edge = torch.sigmoid(self.edge_head(feat))
        root = torch.sigmoid(self.root_head(feat))
        density = root
        centers, pix_coord, ctr_coord = self._centers_from_pool(feat * (0.5 + root))
        root_centers = F.adaptive_avg_pool2d(root, output_size=centers.shape[-2:]).flatten(2)
        edge_centers = F.adaptive_avg_pool2d(edge, output_size=centers.shape[-2:]).flatten(2)

        pix = F.normalize(feat.flatten(2).transpose(1, 2), dim=-1)
        ctr = F.normalize(centers.flatten(2).transpose(1, 2), dim=-1)
        sim = torch.bmm(pix, ctr.transpose(1, 2))
        spatial = torch.cdist(pix_coord, ctr_coord, p=2).pow(2)
        edge_pixels = edge.flatten(2).transpose(1, 2)
        boundary_cost = edge_pixels + edge_centers.squeeze(1).unsqueeze(1)
        root_bonus = root_centers.squeeze(1).unsqueeze(1)
        logits = (
            sim
            - self.compactness * spatial
            - self.lambda_edge * boundary_cost
            + self.lambda_root * root_bonus
        ) / max(self.temperature, 1e-4)
        return self._emit(centers, logits, h, w, density=density)


def build_top_k_rwkv(dim: int = 32, num_heads: int = 4, **kwargs) -> Top_k_rwkv:
    return Top_k_rwkv(dim=dim, num_heads=num_heads, **kwargs)


def build_spv001(in_channels: int = 3, **kwargs) -> SPV001:
    return SPV001(in_channels=in_channels, **kwargs)


def build_spv002(in_channels: int = 3, **kwargs) -> SPV002:
    return SPV002(in_channels=in_channels, **kwargs)


def build_spv003(in_channels: int = 3, **kwargs) -> SPV003:
    return SPV003(in_channels=in_channels, **kwargs)


def build_spv004(in_channels: int = 3, **kwargs) -> SPV004:
    return SPV004(in_channels=in_channels, **kwargs)


def build_spv005(in_channels: int = 3, **kwargs) -> SPV005:
    return SPV005(in_channels=in_channels, **kwargs)


def build_spv006(in_channels: int = 3, **kwargs) -> SPV006:
    return SPV006(in_channels=in_channels, **kwargs)


def build_spv007(in_channels: int = 3, **kwargs) -> SPV007:
    return SPV007(in_channels=in_channels, **kwargs)


@torch.no_grad()
def demo_top_k_rwkv_shape(
    batch: int = 2,
    channels: int = 32,
    height: int = 32,
    width: int = 32,
    num_heads: int = 4,
    device: str = "cpu",
) -> tuple[torch.Size, torch.Size]:
    x = torch.randn(batch, channels, height, width, device=device)
    block = Top_k_rwkv(channels, num_heads=num_heads, enable_wkv_cuda=False).to(device).eval()
    y = block(x)
    print(f"[Top_k_rwkv] input={tuple(x.shape)} output={tuple(y.shape)}")
    return x.shape, y.shape


@torch.no_grad()
def demo_spv001_shape(
    batch: int = 2,
    channels: int = 3,
    height: int = 64,
    width: int = 64,
    device: str = "cpu",
) -> SuperpixelOutput:
    x = torch.randn(batch, channels, height, width, device=device)
    block = SPV001(in_channels=channels, embed_channels=32, n_spixels=64).to(device).eval()
    out = block(x)
    print(
        "[SPV001] "
        f"input={tuple(x.shape)} vertices={tuple(out.vertices.shape)} "
        f"assignment={tuple(out.assignment.shape)} pixel_features={tuple(out.pixel_features.shape)}"
    )
    return out


@torch.no_grad()
def demo_spv002_shape(
    batch: int = 2,
    channels: int = 3,
    height: int = 64,
    width: int = 64,
    device: str = "cpu",
) -> SuperpixelOutput:
    x = torch.randn(batch, channels, height, width, device=device)
    block = SPV002(in_channels=channels, embed_channels=32, n_spixels=64).to(device).eval()
    out = block(x)
    print(
        "[SPV002] "
        f"input={tuple(x.shape)} vertices={tuple(out.vertices.shape)} "
        f"assignment={tuple(out.assignment.shape)} pixel_features={tuple(out.pixel_features.shape)}"
    )
    return out


@torch.no_grad()
def demo_spv003_shape(
    batch: int = 2,
    channels: int = 3,
    height: int = 64,
    width: int = 64,
    device: str = "cpu",
) -> SuperpixelOutput:
    x = torch.randn(batch, channels, height, width, device=device)
    block = SPV003(in_channels=channels, embed_channels=32, n_spixels=64).to(device).eval()
    out = block(x)
    density_shape = None if out.density is None else tuple(out.density.shape)
    print(
        "[SPV003] "
        f"input={tuple(x.shape)} vertices={tuple(out.vertices.shape)} "
        f"assignment={tuple(out.assignment.shape)} pixel_features={tuple(out.pixel_features.shape)} "
        f"density={density_shape}"
    )
    return out


def _print_sp_output(name: str, x: torch.Tensor, out: SuperpixelOutput) -> None:
    density_shape = None if out.density is None else tuple(out.density.shape)
    index_shape = None if out.center_indices is None else tuple(out.center_indices.shape)
    print(
        f"[{name}] "
        f"input={tuple(x.shape)} vertices={tuple(out.vertices.shape)} "
        f"assignment={tuple(out.assignment.shape)} pixel_features={tuple(out.pixel_features.shape)} "
        f"density={density_shape} center_indices={index_shape}"
    )


@torch.no_grad()
def demo_spv004_shape(
    batch: int = 2,
    channels: int = 3,
    height: int = 64,
    width: int = 64,
    device: str = "cpu",
) -> SuperpixelOutput:
    x = torch.randn(batch, channels, height, width, device=device)
    block = SPV004(in_channels=channels, embed_channels=32, n_spixels=64).to(device).eval()
    out = block(x)
    _print_sp_output("SPV004", x, out)
    return out


@torch.no_grad()
def demo_spv005_shape(
    batch: int = 2,
    channels: int = 3,
    height: int = 64,
    width: int = 64,
    device: str = "cpu",
) -> SuperpixelOutput:
    x = torch.randn(batch, channels, height, width, device=device)
    block = SPV005(in_channels=channels, embed_channels=32, n_spixels=64).to(device).eval()
    out = block(x)
    _print_sp_output("SPV005", x, out)
    return out


@torch.no_grad()
def demo_spv006_shape(
    batch: int = 2,
    channels: int = 3,
    height: int = 64,
    width: int = 64,
    device: str = "cpu",
) -> SuperpixelOutput:
    x = torch.randn(batch, channels, height, width, device=device)
    block = SPV006(in_channels=channels, embed_channels=32, n_spixels=64).to(device).eval()
    out = block(x)
    _print_sp_output("SPV006", x, out)
    return out


@torch.no_grad()
def demo_spv007_shape(
    batch: int = 2,
    channels: int = 3,
    height: int = 64,
    width: int = 64,
    device: str = "cpu",
) -> SuperpixelOutput:
    x = torch.randn(batch, channels, height, width, device=device)
    block = SPV007(in_channels=channels, embed_channels=32, n_spixels=64).to(device).eval()
    out = block(x)
    _print_sp_output("SPV007", x, out)
    return out


def demo_all_shapes(device: str = "cpu") -> None:
    demo_top_k_rwkv_shape(device=device)
    demo_spv001_shape(device=device)
    demo_spv002_shape(device=device)
    demo_spv003_shape(device=device)
    demo_spv004_shape(device=device)
    demo_spv005_shape(device=device)
    demo_spv006_shape(device=device)
    demo_spv007_shape(device=device)


if __name__ == "__main__":
    demo_all_shapes()
