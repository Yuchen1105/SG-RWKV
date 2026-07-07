from __future__ import annotations

import torch
import torch.nn.functional as F

from .HS3R_Net_Block import SPV005, SuperpixelOutput, _dense_reconstruct


def _spv005_energy_terms(module: SPV005, x: torch.Tensor):
    feat = module.energy_proj(module.stem(x))
    _, _, h, w = feat.shape
    edge = torch.sigmoid(module.edge_head(feat))
    centers, pix_coord, ctr_coord = module._centers_from_pool(feat)
    edge_centers = F.adaptive_avg_pool2d(edge, output_size=centers.shape[-2:]).flatten(2)

    pix = F.normalize(feat.flatten(2).transpose(1, 2), dim=-1)
    ctr = F.normalize(centers.flatten(2).transpose(1, 2), dim=-1)
    appearance_energy = torch.bmm(pix, ctr.transpose(1, 2)) / max(module.temperature, 1e-4)
    spatial_energy = torch.cdist(pix_coord, ctr_coord, p=2).pow(2)
    edge_pixels = edge.flatten(2).transpose(1, 2)
    boundary_energy = edge_pixels + edge_centers.squeeze(1).unsqueeze(1)
    return centers, h, w, appearance_energy, spatial_energy, boundary_energy


class SPV005NoAppearance(SPV005):
    """SPV005 without the appearance-similarity energy term."""

    def forward(self, x: torch.Tensor) -> SuperpixelOutput:
        centers, h, w, _appearance, spatial, boundary = _spv005_energy_terms(self, x)
        logits = -self.compactness * spatial - self.lambda_edge * boundary
        return self._emit(centers, logits, h, w)


class SPV005NoSpatial(SPV005):
    """SPV005 without the spatial-compactness energy term."""

    def forward(self, x: torch.Tensor) -> SuperpixelOutput:
        centers, h, w, appearance, _spatial, boundary = _spv005_energy_terms(self, x)
        logits = appearance - self.lambda_edge * boundary
        return self._emit(centers, logits, h, w)


class SPV005NoBoundary(SPV005):
    """SPV005 without the boundary-confidence energy term."""

    def forward(self, x: torch.Tensor) -> SuperpixelOutput:
        centers, h, w, appearance, spatial, _boundary = _spv005_energy_terms(self, x)
        logits = appearance - self.compactness * spatial
        return self._emit(centers, logits, h, w)


class SPV005DenseAssignment(SPV005):
    """SPV005 with dense soft assignment over all prototypes."""

    def forward(self, x: torch.Tensor) -> SuperpixelOutput:
        centers, h, w, appearance, spatial, boundary = _spv005_energy_terms(self, x)
        logits = appearance - self.compactness * spatial - self.lambda_edge * boundary
        pixel_features, assignment, center_indices = _dense_reconstruct(centers, logits, h, w)
        return SuperpixelOutput(
            vertices=centers,
            assignment=assignment,
            pixel_features=pixel_features,
            center_indices=center_indices,
        )
