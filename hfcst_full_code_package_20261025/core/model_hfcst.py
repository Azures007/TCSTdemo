"""HFCST-v9 dual-view hyperspectral classifier.

The spatial view is per-pixel L2 normalized; the spectral view retains the
shared-scale amplitude. Each view has its own classifier head and supervision.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int, preferred: int = 8) -> int:
    for g in range(min(channels, preferred), 0, -1):
        if channels % g == 0:
            return g
    return 1


class _Conv3d(nn.Sequential):
    def __init__(self, ci: int, co: int, kernel, stride=1, padding=0):
        super().__init__(
            nn.Conv3d(ci, co, kernel, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_groups(co), co), nn.SiLU(inplace=True))


class _Residual2d(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=dilation,
                            dilation=dilation, bias=False)
        self.n1 = nn.GroupNorm(_groups(channels), channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(_groups(channels), channels)
        self.drop = nn.Dropout2d(0.05)

    def forward(self, x):
        z = F.silu(self.n1(self.c1(x)), inplace=True)
        z = self.n2(self.c2(self.drop(z)))
        return F.silu(x + z, inplace=True)


@dataclass
class HFCSTOutput:
    logits: torch.Tensor
    context_logits: torch.Tensor
    spectral_logits: torch.Tensor
    features: torch.Tensor
    context_features: torch.Tensor
    spectral_features: torch.Tensor
    spectral_reliability: torch.Tensor
    scale_weights: torch.Tensor
    center_weights: torch.Tensor


class HFCSTModel(nn.Module):
    """Compact 3-D spectral-spatial + 2-D multiscale context + spectral encoder."""
    def __init__(self, bands: int, num_classes: int, feature_dim: int = 64,
                 max_gate: float = 0.5, dropout: float = 0.10):
        super().__init__()
        if bands < 3:
            raise ValueError("HFCST requires at least 3 spectral bands")
        if not (0.0 < max_gate <= 1.0):
            raise ValueError("max_gate must be in (0, 1]")
        self.bands = int(bands)
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.max_gate = float(max_gate)

        self.spatial3d = nn.Sequential(
            _Conv3d(1, 16, (7, 3, 3), stride=(2, 1, 1), padding=(3, 1, 1)),
            _Conv3d(16, 24, (5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            _Conv3d(24, 32, (3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)))
        self.context_stem = nn.Sequential(
            nn.Conv2d(32, 48, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(48), 48), nn.SiLU(inplace=True),
            _Residual2d(48, dilation=1))
        self.context_scales = nn.ModuleList([
            nn.Sequential(nn.Conv2d(48, 24, 3, padding=d, dilation=d, bias=False),
                          nn.GroupNorm(_groups(24), 24), nn.SiLU(inplace=True),
                          _Residual2d(24, dilation=1))
            for d in (1, 2, 3)])
        self.scale_selector = nn.Sequential(
            nn.Linear(24 * 3, 48), nn.SiLU(inplace=True), nn.Linear(48, 3))
        self.context_fuse = nn.Sequential(
            nn.Conv2d(72, 64, 1, bias=False), nn.GroupNorm(_groups(64), 64),
            nn.SiLU(inplace=True))
        self.center_projection = nn.Sequential(
            nn.Linear(64 * 3, 128, bias=False), nn.LayerNorm(128),
            nn.SiLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, feature_dim), nn.LayerNorm(feature_dim))

        self.spectral_raw = nn.Sequential(
            nn.LayerNorm(bands), nn.Linear(bands, 96, bias=False),
            nn.LayerNorm(96), nn.SiLU(inplace=True))
        self.spectral_shape = nn.Sequential(
            nn.Conv1d(3, 24, 5, padding=2, bias=False),
            nn.GroupNorm(_groups(24), 24), nn.SiLU(inplace=True),
            nn.Conv1d(24, 32, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(32), 32), nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(1))
        self.spectral_amplitude = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(inplace=True), nn.Linear(16, 16), nn.SiLU(inplace=True))
        self.spectral_projection = nn.Sequential(
            nn.Linear(96 + 32 + 16, 128, bias=False), nn.LayerNorm(128),
            nn.SiLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, feature_dim), nn.LayerNorm(feature_dim))
        self.spectral_to_context = nn.Linear(feature_dim, feature_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 3, 48), nn.SiLU(inplace=True), nn.Linear(48, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -1.5)

        # Independent heads prevent early branch competition through a shared head.
        self.context_head = nn.Linear(feature_dim, num_classes)
        self.spectral_head = nn.Linear(feature_dim, num_classes)
        self.fused_head = nn.Linear(feature_dim, num_classes)
        self._init_weights()
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -1.5)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> HFCSTOutput:
        if x.ndim != 4 or x.shape[1] != self.bands:
            raise ValueError(f"expected [N,{self.bands},H,W], got {tuple(x.shape)}")
        # The spatial view uses shape-only spectra; spectral branch keeps amplitude.
        x_spatial = F.normalize(x.float(), p=2, dim=1, eps=1e-6)
        f3 = self.spatial3d(x_spatial.unsqueeze(1)).mean(dim=2)
        f0 = self.context_stem(f3)
        scales = [branch(f0) for branch in self.context_scales]
        desc = torch.cat([F.adaptive_avg_pool2d(z, 1).flatten(1) for z in scales], dim=1)
        sw = torch.softmax(self.scale_selector(desc), dim=1)
        # Per-sample scale weights broadcast across channel and spatial dimensions.
        weighted = [z * sw[:, i].view(-1, 1, 1, 1) for i, z in enumerate(scales)]
        ctx_map = self.context_fuse(torch.cat(weighted, dim=1))
        global_pool = F.adaptive_avg_pool2d(ctx_map, 1).flatten(1)
        hh, ww = ctx_map.shape[-2:]
        ch, cw = min(3, hh), min(3, ww)
        r0, c0 = max(0, (hh - ch) // 2), max(0, (ww - cw) // 2)
        center = F.adaptive_avg_pool2d(ctx_map[:, :, r0:r0+ch, c0:c0+cw], 1).flatten(1)
        tokens = ctx_map.flatten(2).transpose(1, 2)
        similarity = F.normalize(tokens, dim=2) * F.normalize(center, dim=1)[:, None, :]
        att = torch.softmax(similarity.sum(dim=2) / 0.12, dim=1)
        attended = (tokens * att[:, :, None]).sum(dim=1)
        cwts = torch.softmax(torch.stack([
            F.cosine_similarity(global_pool, center, dim=1),
            F.cosine_similarity(center, attended, dim=1),
            F.cosine_similarity(global_pool, attended, dim=1)], dim=1), dim=1)
        context_features = F.normalize(self.center_projection(
            torch.cat([global_pool, center, attended], dim=1)), dim=1)

        spec = x.float()[:, :, x.shape[2] // 2, x.shape[3] // 2]
        mean = spec.mean(dim=1, keepdim=True)
        std = spec.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        z = (spec - mean) / std
        d1 = F.pad(z[:, 1:] - z[:, :-1], (0, 1))
        d2 = F.pad(d1[:, 1:] - d1[:, :-1], (0, 1))
        shape = self.spectral_shape(torch.stack([z, d1, d2], dim=1)).flatten(1)
        raw = self.spectral_raw(spec)
        log_amp = torch.log(torch.linalg.vector_norm(spec, dim=1, keepdim=True).clamp_min(1e-6))
        spectral_features = F.normalize(self.spectral_projection(
            torch.cat([raw, shape, self.spectral_amplitude(log_amp)], dim=1)), dim=1)
        projected_spectral = self.spectral_to_context(spectral_features)
        gate = self.max_gate * torch.sigmoid(self.gate(torch.cat([
            context_features, spectral_features,
            torch.abs(context_features - spectral_features)], dim=1)))
        fused_features = F.normalize(
            context_features + gate * (projected_spectral - context_features), dim=1)
        context_logits = self.context_head(context_features)
        spectral_logits = self.spectral_head(spectral_features)
        logits = self.fused_head(fused_features)
        return HFCSTOutput(
            logits=logits, context_logits=context_logits, spectral_logits=spectral_logits,
            features=fused_features, context_features=context_features,
            spectral_features=spectral_features, spectral_reliability=gate.squeeze(1),
            scale_weights=sw, center_weights=cwts)
