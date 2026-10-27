"""Backbone: DualReliabilityEncoder (from v7.1, unchanged)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int, preferred: int = 8) -> int:
    for group in range(min(channels, preferred), 0, -1):
        if channels % group == 0:
            return group
    return 1


class ConvNormAct3D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class LocalSpectralSpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvNormAct3D(1, 24, (7, 3, 3), stride=(2, 1, 1), padding=(3, 1, 1))
        self.stage1 = ConvNormAct3D(24, 40, (5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1))
        self.stage2 = ConvNormAct3D(40, 64, (3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))
        self.residual = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(64), 64), nn.SiLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(64), 64),
        )

    def forward(self, x):
        f = self.stage2(self.stage1(self.stem(x.unsqueeze(1))))
        return F.silu(f + self.residual(f), inplace=True)


class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.08):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.dropout = nn.Dropout2d(dropout)
        self.shortcut = (nn.Identity() if in_ch == out_ch else
                         nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                       nn.GroupNorm(_groups(out_ch), out_ch)))

    def forward(self, x):
        r = self.shortcut(x)
        x = F.silu(self.norm1(self.conv1(x)), inplace=True)
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return F.silu(x + r, inplace=True)


class _ASPPBranch(nn.Sequential):
    def __init__(self, in_ch, out_ch, k, dilation):
        padding = 0 if k == 1 else dilation
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, dilation=dilation, padding=padding, bias=False),
            nn.GroupNorm(_groups(out_ch), out_ch), nn.SiLU(inplace=True),
        )


class ScaleSelectiveMicroASPP(nn.Module):
    def __init__(self, in_channels=128, branch_channels=32, mode="adaptive"):
        super().__init__()
        if mode not in {"single", "equal", "adaptive"}:
            raise ValueError("MicroASPP mode must be single, equal or adaptive.")
        self.mode = mode
        specs = [(1, 1), (3, 1), (3, 2), (3, 3)]
        self.branches = nn.ModuleList(
            [_ASPPBranch(in_channels, branch_channels, k, d) for k, d in specs])
        dd = branch_channels * len(self.branches)
        hidden = max(24, dd // 4)
        self.selector = nn.Sequential(
            nn.Linear(dd, hidden), nn.SiLU(inplace=True),
            nn.Linear(hidden, len(self.branches)))
        self.fuse = nn.Sequential(
            nn.Conv2d(dd, in_channels, 1, bias=False),
            nn.GroupNorm(_groups(in_channels), in_channels))
        self.residual_logit = nn.Parameter(torch.tensor(-2.1972246))

    def forward(self, x):
        fs = [b(x) for b in self.branches]
        desc = torch.cat([F.adaptive_avg_pool2d(f, 1).flatten(1) for f in fs], dim=1)
        if self.mode == "adaptive":
            w = torch.softmax(self.selector(desc), dim=1)
        elif self.mode == "equal":
            w = desc.new_full((desc.size(0), len(self.branches)), 1.0 / len(self.branches))
        else:
            w = desc.new_zeros((desc.size(0), len(self.branches)))
            w[:, 1] = 1.0
        weighted = [f * w[:, i].view(-1, 1, 1, 1) for i, f in enumerate(fs)]
        ms = self.fuse(torch.cat(weighted, dim=1))
        rw = torch.sigmoid(self.residual_logit)
        return F.silu(x + rw * ms, inplace=True), w


class CenterGuidedPooling(nn.Module):
    def __init__(self, channels, center_size=3, attention_temperature=0.12, mode="guided"):
        super().__init__()
        if mode not in {"gap", "center", "guided"}:
            raise ValueError("Pooling mode must be gap, center or guided.")
        self.mode = mode
        self.center_size = int(center_size)
        self.temperature = max(float(attention_temperature), 1e-4)
        hidden = max(32, channels // 3)
        self.gate = nn.Sequential(
            nn.Linear(channels * 4, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, 3))

    def forward(self, x):
        b, c, hh, ww = x.shape
        g = F.adaptive_avg_pool2d(x, 1).flatten(1)
        ch = min(self.center_size, hh); cw = min(self.center_size, ww)
        r0 = max(0, (hh - ch) // 2); c0 = max(0, (ww - cw) // 2)
        cen = F.adaptive_avg_pool2d(x[:, :, r0:r0+ch, c0:c0+cw], 1).flatten(1)
        q = F.normalize(cen, dim=1)
        tok = x.flatten(2).transpose(1, 2)
        nt = F.normalize(tok, dim=2)
        sim = torch.sum(nt * q[:, None, :], dim=2)
        att = torch.softmax(sim / self.temperature, dim=1)
        attended = torch.sum(tok * att[:, :, None], dim=1)
        diff = (cen - g).abs()
        gi = torch.cat([g, cen, attended, diff], dim=1)
        if self.mode == "guided":
            weights = torch.softmax(self.gate(gi), dim=1)
        elif self.mode == "gap":
            weights = g.new_zeros((b, 3)); weights[:, 0] = 1.0
        else:
            weights = g.new_zeros((b, 3)); weights[:, 1] = 1.0
        pooled = weights[:, 0:1]*g + weights[:, 1:2]*cen + weights[:, 2:3]*attended
        return pooled, weights, att.view(b, hh, ww)


class CenterSpectralShapeEncoder(nn.Module):
    def __init__(self, bands, projection_dim, amplitude=False):
        super().__init__()
        self.bands = int(bands)
        self.amplitude = bool(amplitude)
        self.raw_projection = nn.Sequential(
            nn.LayerNorm(self.bands), nn.Linear(self.bands, 128, bias=False),
            nn.LayerNorm(128), nn.SiLU(inplace=True))
        self.shape_encoder = nn.Sequential(
            nn.Conv1d(3, 48, 5, padding=2, bias=False), nn.GroupNorm(_groups(48), 48),
            nn.SiLU(inplace=True),
            nn.Conv1d(48, 96, 3, padding=1, bias=False), nn.GroupNorm(_groups(96), 96),
            nn.SiLU(inplace=True), nn.AdaptiveAvgPool1d(1))
        self.amplitude_head = (nn.Sequential(nn.Linear(1, 32), nn.SiLU(inplace=True))
                               if self.amplitude else None)
        self.fuse = nn.Sequential(
            nn.Linear(224 + (32 if self.amplitude else 0), 192, bias=False),
            nn.LayerNorm(192), nn.SiLU(inplace=True), nn.Dropout(0.10),
            nn.Linear(192, projection_dim))

    def forward(self, x):
        h, w = x.shape[-2:]
        spec = x[:, :, h//2, w//2].float()
        m = spec.mean(dim=1, keepdim=True)
        s = spec.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        z = (spec - m) / s
        f1 = F.pad(z[:, 1:] - z[:, :-1], (0, 1))
        f2 = F.pad(f1[:, 1:] - f1[:, :-1], (0, 1))
        shape = self.shape_encoder(torch.stack([z, f1, f2], dim=1)).flatten(1)
        raw = self.raw_projection(spec)
        parts = [raw, shape]
        if self.amplitude_head is not None:
            la = torch.log(spec.norm(dim=1, keepdim=True).clamp_min(1e-6))
            parts.append(self.amplitude_head(la))
        return F.normalize(self.fuse(torch.cat(parts, dim=1)), dim=1)


class CosineClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes, initial_scale=12.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        self.log_scale = nn.Parameter(torch.tensor(math.log(initial_scale)))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, f):
        f = F.normalize(f, dim=1)
        w = F.normalize(self.weight, dim=1)
        s = self.log_scale.exp().clamp(4.0, 30.0)
        return s * (f @ w.t())


class SpectralReliabilityFusion(nn.Module):
    def __init__(self, feature_dim, mode="reliable", max_gate=0.16):
        super().__init__()
        self.mode = mode
        self.max_gate = max_gate
        hidden = max(32, feature_dim // 2)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden), nn.SiLU(inplace=True),
            nn.Linear(hidden, feature_dim))
        nn.init.zeros_(self.gate[-1].weight)
        init = 0.12 / max_gate
        nn.init.constant_(self.gate[-1].bias, math.log(init / (1 - init)))

    def forward(self, cf, sf, cl, sl):
        if self.mode == "off":
            return cf, cf.new_zeros(cf.shape), cf.new_zeros(cf.size(0))
        if self.mode == "average":
            f = F.normalize(0.5*cf + 0.5*sf, dim=1)
            return f, cf.new_full(cf.shape, 0.5), cf.new_ones(cf.size(0))
        diff = (cf - sf).abs()
        rg = self.max_gate * torch.sigmoid(
            self.gate(torch.cat([cf, sf, diff], dim=1)))
        cos = torch.sum(F.normalize(cf, dim=1) * F.normalize(sf, dim=1), dim=1)
        ra = torch.sigmoid((cos - 0.20) / 0.10)
        cp = F.softmax(cl.detach().float(), dim=1)
        sp = F.softmax(sl.detach().float(), dim=1)
        sa = cp.argmax(1).eq(sp.argmax(1)).float()
        sc = sp.max(1).values
        cc = cp.max(1).values
        adv = (sc - cc).clamp(0, 1)
        rescue = (1 - sa) * adv * (sc >= 0.55).float()
        sem = 0.05 + 0.95 * sa + 0.55 * rescue
        rel = (0.35 + 0.65 * ra) * sem
        gate = rg * rel[:, None]
        fused = F.normalize(cf + gate * (sf - cf), dim=1)
        return fused, gate, rel


@dataclass
class ModelOutput:
    logits: torch.Tensor
    features: torch.Tensor
    scale_weights: torch.Tensor
    center_weights: torch.Tensor
    attention_map: torch.Tensor
    context_logits: torch.Tensor
    spectral_logits: torch.Tensor
    spectral_reliability: torch.Tensor


class DualReliabilityEncoder(nn.Module):
    def __init__(self, bands, num_classes, projection_dim=128, projection_norm="batch",
                 aspp_mode="adaptive", pooling_mode="guided",
                 spectral_fusion_mode="reliable", amplitude_channel=False):
        super().__init__()
        self.bands = int(bands)
        self.num_classes = int(num_classes)
        self.local3d = LocalSpectralSpatialEncoder()
        self.semantic2d = nn.Sequential(
            ResidualBlock2D(64, 96, dropout=0.06),
            ResidualBlock2D(96, 128, dropout=0.08),
            ResidualBlock2D(128, 128, dropout=0.08))
        self.micro_aspp = ScaleSelectiveMicroASPP(128, 32, mode=aspp_mode)
        self.pool = CenterGuidedPooling(128, center_size=3,
                                        attention_temperature=0.12, mode=pooling_mode)
        norm = (nn.BatchNorm1d(256) if projection_norm == "batch"
                else nn.LayerNorm(256))
        self.projection = nn.Sequential(
            nn.Linear(128, 256, bias=False), norm, nn.SiLU(inplace=True),
            nn.Dropout(0.10), nn.Linear(256, projection_dim))
        self.spectral_encoder = CenterSpectralShapeEncoder(
            self.bands, projection_dim, amplitude=amplitude_channel)
        self.classifier = CosineClassifier(projection_dim, self.num_classes)
        self.spectral_fusion = SpectralReliabilityFusion(
            projection_dim, mode=spectral_fusion_mode)
        self._init()
        # A7 fix: _init() overrides the fusion gate's specialized init.
        # Re-apply the intended init so gate starts near-zero weight (fusion ~= spatial).
        import math as _math
        with __import__('torch').no_grad():
            self.spectral_fusion.gate[-1].weight.zero_()
            _init_val = 0.12 / self.spectral_fusion.max_gate
            self.spectral_fusion.gate[-1].bias.fill_(
                _math.log(_init_val / (1 - _init_val)))

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        f3 = self.local3d(x)
        s = f3.mean(dim=2)
        s = self.semantic2d(s)
        s, sw = self.micro_aspp(s)
        pooled, cw, am = self.pool(s)
        cf = F.normalize(self.projection(pooled), dim=1)
        sf = self.spectral_encoder(x)
        cl = self.classifier(cf)
        sl = self.classifier(sf)
        feats, _, sr = self.spectral_fusion(cf, sf, cl, sl)
        return ModelOutput(
            logits=self.classifier(feats), features=feats,
            scale_weights=sw, center_weights=cw, attention_map=am,
            context_logits=cl, spectral_logits=sl, spectral_reliability=sr)
