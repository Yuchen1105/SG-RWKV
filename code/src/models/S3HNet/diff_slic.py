#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from typing import Tuple, Optional
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# =============================================================================
# Helper Functions
# =============================================================================

def compute_stride_and_padding(img_shape: Tuple[int, int], spixel_shape: Tuple[int, int]):
    """
    Compute stride and padding to ensure the image can be divided into superpixels.
    """
    H, W = img_shape
    Hs, Ws = spixel_shape
    stride_h = (H + Hs - 1) // Hs
    stride_w = (W + Ws - 1) // Ws
    H_pad = stride_h * Hs
    W_pad = stride_w * Ws
    pad_y = H_pad - H
    pad_x = W_pad - W
    return (stride_h, stride_w), (pad_x, pad_y)


def spixel_upsampling(x: torch.Tensor, assignments: torch.Tensor, stride: Optional[Tuple[int, int]] = None,
                      candidate_radius: int = 1) -> torch.Tensor:
    """
    Upsample superpixel features to pixel space using assignment weights.
    """
    B, K, H, W = assignments.shape
    _, C, Hs, Ws = x.shape

    neighbor_range = candidate_radius * 2 + 1
    if stride is None:
        stride, padding = compute_stride_and_padding((H, W), (Hs, Ws))
    else:
        sh, sw = stride
        pad_y = Hs * sh - H
        pad_x = Ws * sw - W
        padding = (pad_x, pad_y)

    sh, sw = stride
    pad_x, pad_y = padding
    assignments_pad = F.pad(assignments, (0, pad_x, 0, pad_y))
    H_pad = H + pad_y
    W_pad = W + pad_x

    n_spixels = Hs * Ws
    P = sh * sw

    cand = F.unfold(x, kernel_size=neighbor_range, padding=candidate_radius)
    cand = cand.view(B, C, K, n_spixels)

    asg = F.unfold(assignments_pad, kernel_size=(sh, sw), stride=(sh, sw))
    asg = asg.view(B, K, P, n_spixels)

    # Weighted sum: Pixel_Feature = Sum(Assignment_Weight * Cluster_Feature)
    up = torch.einsum('bckm,bkpm->bcpm', cand, asg)
    up = up.contiguous().view(B * C, P, n_spixels)
    up = F.fold(up, output_size=(H_pad, W_pad), kernel_size=(sh, sw), stride=(sh, sw))
    up = up.view(B, C, H_pad, W_pad)

    # Crop padding
    return up[:, :, :H, :W]


def spixel_downsampling(x: torch.Tensor, assignments: torch.Tensor, stride: Tuple[int, int] = None,
                        candidate_radius: int = 1) -> torch.Tensor:
    """
    Downsample pixel features to superpixel space.
    """
    batch, _, height_s, width_s = assignments.shape
    height, width = x.shape[-2:]
    channels = x.shape[1]
    if stride is None:
        stride, padding = compute_stride_and_padding((height, width), (height_s, width_s))
    else:
        _, padding = compute_stride_and_padding((height, width), (height_s, width_s))

    pad_x, pad_y = padding
    x = F.pad(x, (0, pad_x, 0, pad_y))
    neighbor_range = candidate_radius * 2 + 1
    kernel_size = (stride[0] * neighbor_range, stride[1] * neighbor_range)
    padding_val = (stride[0] * candidate_radius, stride[1] * candidate_radius)
    n_candidate_pixels = kernel_size[0] * kernel_size[1]

    unfold_elem_feats = F.unfold(x, kernel_size, stride=stride, padding=padding_val)
    unfold_elem_feats = unfold_elem_feats.reshape(batch, channels, n_candidate_pixels, height_s, width_s)
    downsampled_features = torch.einsum('bphw,bcphw->bchw', (assignments, unfold_elem_feats))
    return downsampled_features


# =============================================================================
# DiffSLIC Class
# =============================================================================

class DiffSLIC(nn.Module):
    def __init__(self,
                 n_spixels: int,
                 n_iter: int = 5,
                 tau: float = 0.01,
                 candidate_radius: int = 1,
                 normalize: bool = True,
                 stable: bool = False,
                 sim_type: str = "slic",
                 compactness: float = 0.17,
                 ignore_black_background: bool = True,
                 background_threshold: float = 0.05,
                 coarse_scale: int = 4,
                 lambda_edge: float = 0.5) -> None:
        super().__init__()
        self.n_spixels = n_spixels
        self.n_iter = n_iter
        self.tau = tau
        self.candidate_radius = candidate_radius
        self.normalize = normalize
        self.stable = stable
        self.sim_type = sim_type
        self.compactness = compactness
        self.eps = 1e-8
        self.ignore_black_background = ignore_black_background
        self.background_threshold = background_threshold
        self.coarse_scale = coarse_scale
        self.lambda_edge = lambda_edge

        # Cache for spatial distance map
        self._spatial_dist_cache = {}

    def _get_spatial_dist_map(self, kernel_size, stride, device, dtype):
        """
        Pre-compute spatial distance (XY) for the kernel window.
        """
        key = (kernel_size, stride, device, dtype)
        if key in self._spatial_dist_cache:
            return self._spatial_dist_cache[key]

        kh, kw = kernel_size
        cy, cx = (kh - 1) / 2.0, (kw - 1) / 2.0

        y = torch.arange(kh, device=device, dtype=dtype)
        x = torch.arange(kw, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

        dist2 = (grid_y - cy) ** 2 + (grid_x - cx) ** 2
        dist2 = dist2.reshape(1, 1, -1, 1, 1)

        self._spatial_dist_cache[key] = dist2
        return dist2

    def _run_slic_iterations(self, x: torch.Tensor, edge_map: Optional[torch.Tensor] = None,
                             clst_feats: Optional[torch.Tensor] = None, return_assign: bool = False):
        """
        Core SLIC iteration loop.
        """
        B, C, H, W = x.shape

        # 1. Initialize Centroids
        if clst_feats is None:
            height_s = int(math.sqrt(self.n_spixels * H / W))
            width_s = int(math.sqrt(self.n_spixels * W / H))
            height_s = max(1, height_s)
            width_s = max(1, width_s)

            stride_h = (H + height_s - 1) // height_s
            stride_w = (W + width_s - 1) // width_s
            stride = (stride_h, stride_w)
            clst_feats = F.interpolate(x, size=(height_s, width_s), mode="bilinear", align_corners=False)
        else:
            height_s, width_s = clst_feats.shape[-2:]
            stride = ((H + height_s) // height_s, (W + width_s) // width_s)

        if self.normalize and self.sim_type != 'slic':
            x = x / (x.norm(dim=1, keepdim=True) + self.eps)
            clst_feats = clst_feats / (clst_feats.norm(dim=1, keepdim=True) + self.eps)

        # 2. Prepare Padding
        pad_x = (width_s - W % width_s) % width_s
        pad_y = (height_s - H % height_s) % height_s
        x_pad = F.pad(x, (0, pad_x, 0, pad_y))

        # Handle Edge Map Padding and Downsampling
        edge_map_pad = None
        edge_map_down = None
        if edge_map is not None:
            edge_map_pad = F.pad(edge_map, (0, pad_x, 0, pad_y))
            # Downsample to approximate edge strength at cluster center locations
            edge_map_down = F.interpolate(edge_map, size=(height_s, width_s), mode='bilinear', align_corners=False)

        # Handle Black Background Mask
        valid_mask = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
        if self.ignore_black_background:
            pixel_magnitude = x.abs().mean(dim=1, keepdim=True)
            content_mask = (pixel_magnitude > self.background_threshold).float()
            valid_mask = valid_mask * content_mask
        valid_mask_pad = F.pad(valid_mask, (0, pad_x, 0, pad_y))

        # 3. Pre-computation (Unfold)
        neighbor_range = self.candidate_radius * 2 + 1
        kernel_size = (stride[0] * neighbor_range, stride[1] * neighbor_range)
        padding = (stride[0] * self.candidate_radius, stride[1] * self.candidate_radius)

        x_unfolded = F.unfold(x_pad, kernel_size, padding=padding, stride=stride)
        n_candidate_pixels = kernel_size[0] * kernel_size[1]
        x_unfolded = x_unfolded.view(B, C, n_candidate_pixels, height_s, width_s)

        mask_unfolded = F.unfold(valid_mask_pad, kernel_size, padding=padding, stride=stride)
        mask_unfolded = mask_unfolded.view(B, 1, n_candidate_pixels, height_s, width_s)
        invalid_neighbor_mask = (mask_unfolded < 0.5)

        edge_unfolded = None
        if edge_map_pad is not None:
            edge_unfolded = F.unfold(edge_map_pad, kernel_size, padding=padding, stride=stride)
            edge_unfolded = edge_unfolded.view(B, 1, n_candidate_pixels, height_s, width_s)

        # Spatial Distance Weights
        dist_xy = None
        if self.sim_type == 'slic':
            S = float(max(stride))
            lam_sq = (self.compactness / (S + self.eps)) ** 2
            dist_xy = self._get_spatial_dist_map(kernel_size, stride, x.device, x.dtype)
            dist_xy = dist_xy * lam_sq

        # 4. Iteration Loop
        soft_assign = None
        for _ in range(self.n_iter):
            c_expanded = clst_feats.unsqueeze(2)  # (B, C, 1, Hs, Ws)

            # Feature Distance
            diff = x_unfolded - c_expanded
            dist = (diff * diff).sum(dim=1, keepdim=True)

            if self.sim_type == 'slic':
                dist = dist + dist_xy

            # Boundary Barrier Term
            if edge_unfolded is not None and self.lambda_edge > 0:
                c_edge_expanded = edge_map_down.unsqueeze(2)
                # Path Barrier approx. = Avg(Pixel_Edge, Center_Edge)
                barrier = (edge_unfolded + c_edge_expanded) * 0.5
                dist = dist + self.lambda_edge * barrier

            similarities = -dist
            similarities = similarities.masked_fill(invalid_neighbor_mask, -1e9)

            if self.stable:
                similarities = similarities - similarities.amax(dim=2, keepdim=True).detach()

            soft_assign = F.softmax(similarities / self.tau, dim=2)

            # M-Step: Update Centers
            new_clst_feats = (x_unfolded * soft_assign).sum(dim=2)
            clst_feats = new_clst_feats
            if self.normalize and self.sim_type != 'slic':
                clst_feats = clst_feats / (clst_feats.norm(dim=1, keepdim=True) + self.eps)

        if return_assign:
            return clst_feats, soft_assign.squeeze(1)

        return clst_feats

    def forward(self, x: torch.Tensor, clst_feats: Optional[torch.Tensor] = None,
                edge_logits: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Coarse-to-Fine Forward Pass with Boundary Awareness
        """
        B, C, H, W = x.shape

        # Handle Edge Logits
        edge_map = None
        if edge_logits is not None:
            edge_map = torch.sigmoid(edge_logits)

        # Stage 1: Coarse Scale Iteration
        if self.coarse_scale > 1 and clst_feats is None:
            # Downsample Image
            x_small = F.interpolate(x, scale_factor=1.0 / self.coarse_scale, mode='bilinear', align_corners=False)

            # Downsample Edge Map
            edge_small = None
            if edge_map is not None:
                edge_small = F.interpolate(edge_map, scale_factor=1.0 / self.coarse_scale, mode='bilinear',
                                           align_corners=False)

            # Run iterations on coarse scale
            clst_feats = self._run_slic_iterations(x_small, edge_map=edge_small)

        elif clst_feats is None:
            clst_feats = None

        if clst_feats is None:  # Run full resolution if no downsampling
            clst_feats = self._run_slic_iterations(x, edge_map=edge_map)

        # Stage 2: Fine Refinement (Full Resolution)
        height_s, width_s = clst_feats.shape[-2:]
        stride_h = (H + height_s - 1) // height_s
        stride_w = (W + width_s - 1) // width_s
        stride = (stride_h, stride_w)

        pad_x = (width_s - W % width_s) % width_s
        pad_y = (height_s - H % height_s) % height_s

        x_pad = F.pad(x, (0, pad_x, 0, pad_y))

        edge_map_pad = None
        if edge_map is not None:
            edge_map_pad = F.pad(edge_map, (0, pad_x, 0, pad_y))

        # Create Background Mask
        valid_mask = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
        if self.ignore_black_background:
            pixel_magnitude = x.abs().mean(dim=1, keepdim=True)
            content_mask = (pixel_magnitude > self.background_threshold).float()
            valid_mask = valid_mask * content_mask
        valid_mask_pad = F.pad(valid_mask, (0, pad_x, 0, pad_y))

        # Final Assignment
        p2s_assign, similarities = self.compute_elem_to_center_assignment(
            clst_feats, x_pad, stride, valid_mask_pad, edge_map_pad=edge_map_pad
        )

        # Crop Padding
        if pad_y > 0: p2s_assign = p2s_assign[..., :-pad_y, :]
        if pad_x > 0: p2s_assign = p2s_assign[..., :-pad_x]

        s2p_assign = None  # Legacy output placeholder

        return clst_feats, p2s_assign, s2p_assign

    def compute_elem_to_center_assignment(self, clst_feats, elem_feats, stride=None, mask_pad=None, edge_map_pad=None):
        """
        Calculates pixel-to-center assignment (Soft Assignment).
        Automatically handles padding for mismatched dimensions and incorporates boundary barriers.
        """
        B, C, H, W = elem_feats.shape
        Hs, Ws = clst_feats.shape[-2:]
        neighbor_range = self.candidate_radius * 2 + 1
        K = neighbor_range ** 2

        # 1. Auto-calculate Stride
        if stride is None:
            stride_h = (H + Hs - 1) // Hs
            stride_w = (W + Ws - 1) // Ws
            stride = (max(1, stride_h), max(1, stride_w))

        sh, sw = stride

        # 2. Auto-padding Calculation
        H_target = Hs * sh
        W_target = Ws * sw
        pad_h = max(0, H_target - H)
        pad_w = max(0, W_target - W)

        # Pad Features
        if pad_h > 0 or pad_w > 0:
            elem_feats_pad = F.pad(elem_feats, (0, pad_w, 0, pad_h))
        else:
            elem_feats_pad = elem_feats

        # Pad Mask
        if mask_pad is None:
            mask_pad = torch.ones(B, 1, H, W, device=elem_feats.device, dtype=elem_feats.dtype)
            if pad_h > 0 or pad_w > 0:
                mask_pad = F.pad(mask_pad, (0, pad_w, 0, pad_h))
        elif mask_pad.shape[-2:] != (H_target, W_target):
            mask_pad = F.pad(mask_pad, (0, pad_w, 0, pad_h))

        # Pad Edge Map
        if edge_map_pad is not None:
            if edge_map_pad.shape[-2:] != (H_target, W_target):
                edge_map_pad = F.pad(edge_map_pad, (0, pad_w, 0, pad_h))

        # 3. Unfold Centers
        candidate_clusters = F.unfold(clst_feats, kernel_size=neighbor_range, padding=self.candidate_radius)
        candidate_clusters = candidate_clusters.reshape(B, C, K, Hs * Ws)

        # 4. Unfold Pixels
        unfold_elem_feats = F.unfold(elem_feats_pad, kernel_size=(sh, sw), stride=(sh, sw))
        P = sh * sw
        unfold_elem_feats = unfold_elem_feats.reshape(B, C, P, Hs * Ws)

        # 5. Calculate Distance
        diff = unfold_elem_feats.unsqueeze(2) - candidate_clusters.unsqueeze(3)
        dist = (diff.pow(2)).sum(1)

        # Add Spatial Distance
        if self.sim_type == 'slic':
            device = elem_feats.device
            py, px = torch.meshgrid(torch.arange(sh, device=device), torch.arange(sw, device=device), indexing='ij')
            py, px = py.reshape(1, 1, P, 1), px.reshape(1, 1, P, 1)

            r = self.candidate_radius
            oy, ox = torch.meshgrid(torch.arange(-r, r + 1, device=device), torch.arange(-r, r + 1, device=device),
                                    indexing='ij')
            oy, ox = oy.reshape(1, K, 1, 1), ox.reshape(1, K, 1, 1)

            cy_blk, cx_blk = (sh - 1) / 2.0, (sw - 1) / 2.0
            cen_y = oy * float(sh) + cy_blk
            cen_x = ox * float(sw) + cx_blk

            dist_xy = (py - cen_y) ** 2 + (px - cen_x) ** 2

            S = float(max(stride))
            lam_sq = (self.compactness / (S + self.eps)) ** 2
            dist = dist + lam_sq * dist_xy

        # Add Boundary Barrier
        if edge_map_pad is not None and self.lambda_edge > 0:
            # Pixel Edge: Unfold
            unfold_edge_pix = F.unfold(edge_map_pad, kernel_size=(sh, sw), stride=(sh, sw))
            unfold_edge_pix = unfold_edge_pix.reshape(B, 1, P, Hs * Ws)

            # Center Edge: Downsample & Unfold
            edge_down = F.interpolate(edge_map_pad, size=(Hs, Ws), mode='bilinear', align_corners=False)
            candidate_edge_centers = F.unfold(edge_down, kernel_size=neighbor_range, padding=self.candidate_radius)
            candidate_edge_centers = candidate_edge_centers.reshape(B, 1, K, Hs * Ws)

            # Barrier Calculation
            barrier = (unfold_edge_pix.unsqueeze(2) + candidate_edge_centers.unsqueeze(3)) * 0.5
            dist = dist + self.lambda_edge * barrier.squeeze(1)

        similarities = -dist

        # 6. Apply Mask
        unfold_mask = F.unfold(mask_pad, kernel_size=(sh, sw), stride=(sh, sw))
        unfold_mask = unfold_mask.reshape(B, 1, P, Hs * Ws)
        pixel_invalid = (unfold_mask < 0.5)
        similarities = similarities.masked_fill(pixel_invalid, -1e9)

        if self.stable:
            similarities = similarities - similarities.amax(1, keepdim=True).detach()

        soft_assignment = torch.softmax(similarities / self.tau, dim=1)

        # 7. Fold (Restore to image size)
        soft_assignment = soft_assignment.view(B, K * P, Hs * Ws)
        soft_assignment = F.fold(soft_assignment, output_size=(H_target, W_target), kernel_size=(sh, sw),
                                 stride=(sh, sw))

        # Crop Padding
        if pad_h > 0: soft_assignment = soft_assignment[..., :-pad_h, :]
        if pad_w > 0: soft_assignment = soft_assignment[..., :-pad_w]

        return soft_assignment, similarities


def lbl_to_rgb(lbl: np.ndarray, color_palette: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    height, width = lbl.shape[-2:]
    if color_palette is None:
        n_segments = height * width
        color_palette = np.random.randint(0, 255, (n_segments, 3), dtype=np.uint8)
    return color_palette[lbl], color_palette


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries
    from skimage.segmentation.slic_superpixels import _enforce_label_connectivity_cython

    # Setup
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_spix', default=256, type=int)
    parser.add_argument('--n_iter', default=5, type=int)
    parser.add_argument('--tau', default=0.01, type=float)  # Temperature for softmax
    parser.add_argument('--stable', action='store_true')
    parser.add_argument('--candidate_radius', default=1, type=int)
    # Mock args for notebook execution
    args = parser.parse_args(args=[])

    # Load Image
    try:
        image_path = "test_slic.png"
        img_pil = Image.open(image_path).convert('RGB')
    except:
        # Generate dummy image if file not found
        img_pil = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        print("Warning: slic_test.jpg not found, using random noise.")

    img_np = np.array(img_pil)
    img = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).contiguous().float() / 255 * 2 - 1
    h, w = img.shape[-2:]

    coords = torch.stack(torch.meshgrid(torch.linspace(-1, 1, h, dtype=torch.float),
                                        torch.linspace(-1, 1, w, dtype=torch.float), indexing='ij'), -1).unsqueeze(0)

    # sin embedding
    freqs = 2 ** torch.arange(2, dtype=torch.float)
    shape = coords.shape[:-1] + (-1,)
    scaled_x = (coords[..., None, :] * freqs[..., None]).reshape(shape)  # (batch, *, n_points, num_feats * n_freq)
    scaled_x = torch.stack([scaled_x, scaled_x + 0.5 * torch.pi], -2).reshape(
        shape)  # (batch, n_points, 2 * num_feats * n_freq)
    embedded_x = torch.sin(scaled_x).permute(0, 3, 1, 2) * 1.0

    inputs = img.cuda()

    print(f"Running DiffSLIC with n_spix={args.n_spix}, iter={args.n_iter}...")
    model = DiffSLIC(
        args.n_spix,
        args.n_iter,
        args.tau,
        args.candidate_radius,
        stable=args.stable,
        sim_type='slic',
        compactness=0.17,
        coarse_scale=4,
        ignore_black_background=True
    ).cuda()

    # Warmup
    model(inputs)

    # Timing
    torch.cuda.synchronize()
    starter.record()
    feats, assign, p2s = model(inputs)
    ender.record()
    torch.cuda.synchronize()
    print(f"DiffSLIC Time: {starter.elapsed_time(ender):.2f} ms")

    # assignment to label
    hard_assign = F.one_hot(assign.argmax(1), (2 * args.candidate_radius + 1) ** 2).permute(0, 3, 1,
                                                                                            2).contiguous().float()

    # -----------------------------------------------------------
    # Visualization Logic
    # -----------------------------------------------------------
    h_s, w_s = feats.shape[-2:]

    # Create a grid of global IDs (0 ... Hs*Ws-1)
    # shape (1, 1, Hs, Ws)
    label_grid = torch.arange(h_s * w_s, dtype=torch.float, device=inputs.device).reshape(1, 1, h_s, w_s)

    global_label_map = spixel_upsampling(label_grid, hard_assign, candidate_radius=args.candidate_radius)
    np_lbl = global_label_map[0, 0].cpu().numpy().astype(np.int64)

    # enforce connectivity
    segment_size = h * w / (h_s * w_s)
    min_size = int(0.06 * segment_size)
    max_size = int(3.0 * segment_size)
    np_lbl = _enforce_label_connectivity_cython(np_lbl[None], min_size, max_size)[0]
    valid_n_spixel = len(np.unique(np_lbl))
    print(f"#Superpixels {valid_n_spixel}")

    # 保存 Post-processed Boundaries 图像
    fig3 = plt.figure(figsize=(12, 12))
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.imshow(mark_boundaries(img_np, np_lbl))
    ax3.axis('off')
    plt.savefig("post_processed_boundaries.png", dpi=300, bbox_inches='tight')
    plt.show()
    # plt.close(fig3)

    # 保存 Post-processed Colors 图像
    fig4 = plt.figure(figsize=(12, 12))
    ax4 = fig4.add_subplot(1, 1, 1)
    rgb_lbl, color_palette = lbl_to_rgb(np_lbl)
    ax4.imshow(rgb_lbl)
    ax4.axis('off')
    plt.savefig("post_processed_colors.png", dpi=300, bbox_inches='tight')
    plt.show()
    # plt.close(fig4)
