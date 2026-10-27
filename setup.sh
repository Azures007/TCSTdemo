#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p core configs scripts tests results

############################## requirements.txt ##############################
cat > requirements.txt << 'EOF'
torch>=2.0
numpy>=1.24
scipy>=1.10
h5py>=3.8
pyyaml>=6.0
cvxpy>=1.3
tqdm>=4.65
EOF

############################## core/__init__.py ##############################
cat > core/__init__.py << 'EOF'
"""T-CST v8.0 Core."""
EOF

############################## core/model.py ##############################
cat > core/model.py << 'EOF'
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
        b, c, h, w = x.shape
        g = F.adaptive_avg_pool2d(x, 1).flatten(1)
        ch = min(self.center_size, h); cw = min(self.center_size, w)
        r0 = max(0, (h - ch) // 2); c0 = max(0, (w - cw) // 2)
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
            w = torch.softmax(self.gate(gi), dim=1)
        elif self.mode == "gap":
            w = g.new_zeros((b, 3)); w[:, 0] = 1.0
        else:
            w = g.new_zeros((b, 3)); w[:, 1] = 1.0
        pooled = w[:, 0:1]*g + w[:, 1:2]*cen + w[:, 2:3]*attended
        return pooled, w, att.view(b, h, w)


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
EOF

############################## core/spectral_transform.py ##############################
cat > core/spectral_transform.py << 'EOF'
"""Low-dim smooth spectral transform (DCT gain + offset)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def build_dct_basis(bands, K=8, device=None, dtype=torch.float32):
    if K <= 0 or K > bands:
        raise ValueError(f"K={K} must be in (0, bands={bands}]")
    k = np.arange(K)[None, :]
    i = np.arange(bands)[:, None]
    B = np.cos(np.pi * (2 * i + 1) * k / (2 * bands)) * math.sqrt(2.0 / bands)
    B[:, 0] *= 1.0 / math.sqrt(2.0)
    return torch.from_numpy(B.astype(np.float32)).to(device=device, dtype=dtype)


class SmoothSpectralTransform(nn.Module):
    def __init__(self, bands, K=8, rho_a=0.10, rho_b=0.10):
        super().__init__()
        self.bands = int(bands)
        self.K = int(K)
        self.rho_a = float(rho_a)
        self.rho_b = float(rho_b)
        self.register_buffer("U", build_dct_basis(self.bands, self.K), persistent=True)

    @property
    def dim(self):
        return 2 * self.K

    def gain(self, alpha):
        return torch.exp(self.rho_a * torch.tanh(self.U @ alpha))

    def offset(self, beta):
        return self.rho_b * torch.tanh(self.U @ beta)

    def forward(self, x, phi):
        if x.shape[1] != self.bands:
            raise ValueError(f"Expected {self.bands} bands, got {x.shape[1]}")
        a = self.gain(phi[: self.K]).view(1, -1, 1, 1)
        b = self.offset(phi[self.K:]).view(1, -1, 1, 1)
        return x * a + b

    @torch.no_grad()
    def diagnostics(self, x, phi):
        xt = self.forward(x, phi)
        diff = xt - x
        rel = float(diff.norm() / x.norm().clamp_min(1e-8))
        a = self.gain(phi[: self.K])
        b = self.offset(phi[self.K:])
        return {"relative_spectral_change": rel,
                "gain_min": float(a.min()), "gain_max": float(a.max()),
                "offset_min": float(b.min()), "offset_max": float(b.max())}
EOF

############################## core/discrepancy.py ##############################
cat > core/discrepancy.py << 'EOF'
"""Spectral descriptor, MK-MMD, SWD, domain gradient."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def spectral_descriptor(x, eps=1e-6):
    N, B, H, W = x.shape
    mu = x.mean(dim=(-2, -1))
    sigma = x.std(dim=(-2, -1), unbiased=False)
    log_sigma = torch.log(sigma + eps)
    d1 = mu[:, 1:] - mu[:, :-1]
    d2 = d1[:, 1:] - d1[:, :-1]
    return torch.cat([mu, log_sigma, d1, d2], dim=1)


def _sq_dist(x, y):
    xx = x.pow(2).sum(1, keepdim=True)
    yy = y.pow(2).sum(1, keepdim=True)
    return (xx + yy.t() - 2 * (x @ y.t())).clamp_min(0.0)


def mk_mmd2(x, y, bandwidths=(1.0, 2.0, 4.0, 8.0)):
    dxx = _sq_dist(x, x); dyy = _sq_dist(y, y); dxy = _sq_dist(x, y)
    total = x.new_tensor(0.0)
    for bw in bandwidths:
        total = total + (torch.exp(-dxx / (2*bw*bw)).mean()
                         + torch.exp(-dyy / (2*bw*bw)).mean()
                         - 2 * torch.exp(-dxy / (2*bw*bw)).mean())
    return total / len(bandwidths)


def domain_gradient_mmd(transform, source_images, target_descriptor,
                        bandwidths=(1.0, 2.0, 4.0, 8.0)):
    phi = torch.zeros(transform.dim, device=source_images.device, requires_grad=True)
    d_s = spectral_descriptor(transform(source_images, phi))
    loss = mk_mmd2(d_s, target_descriptor, bandwidths)
    grad = torch.autograd.grad(loss, phi, create_graph=False)[0]
    return grad.detach(), float(loss.detach())


@torch.no_grad()
def sliced_wasserstein(x, y, num_projections=64, seed=0):
    device = x.device
    g = torch.Generator(device=device).manual_seed(int(seed))
    D = x.shape[1]
    proj = F.normalize(torch.randn(D, num_projections, generator=g, device=device), dim=0)
    px, _ = torch.sort(x @ proj, dim=0)
    py, _ = torch.sort(y @ proj, dim=0)
    return float((px - py).abs().mean())
EOF

############################## core/transport_solver.py ##############################
cat > core/transport_solver.py << 'EOF'
"""Margin-constrained transport QP."""
from __future__ import annotations

import numpy as np


def solve_margin_qp(g_D, G, margins0, weights, rho_m=0.05,
                    lambda_d=1.0, lambda_xi=10.0, verbose=False):
    import cvxpy as cp

    g_D = np.asarray(g_D, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    margins0 = np.asarray(margins0, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    dim = g_D.shape[0]
    J = G.shape[0]

    if J == 0:
        d_star = (-g_D / max(lambda_d, 1e-8)).astype(np.float32)
        return {"status": "no_constraints", "d": d_star,
                "directional_derivative": float(g_D @ d_star),
                "d_norm": float(np.linalg.norm(d_star)),
                "slack_mean": 0.0, "slack_max": 0.0,
                "violation_max": 0.0, "num_constraints": 0}

    eps = rho_m * np.clip(margins0, 0.0, None)
    d = cp.Variable(dim); xi = cp.Variable(J, nonneg=True)
    W = cp.diag(np.clip(weights, 1e-8, None))
    objective = cp.Minimize(g_D @ d + (lambda_d/2) * cp.sum_squares(d)
                            + (lambda_xi/2) * cp.quad_form(xi, W))
    constraints = [G @ d >= -eps - xi]
    prob = cp.Problem(objective, constraints)

    solved = False
    for solver in (cp.ECOS, cp.OSQP):
        try:
            prob.solve(solver=solver, verbose=verbose, warm_start=True)
            solved = True
            break
        except Exception:
            continue

    if not solved or d.value is None:
        return {"status": "solver_failure",
                "d": np.zeros(dim, dtype=np.float32),
                "directional_derivative": 0.0, "d_norm": 0.0,
                "slack_mean": 0.0, "slack_max": 0.0,
                "violation_max": float("inf"), "num_constraints": int(J)}

    d_star = d.value.astype(np.float32)
    Gt = G @ d_star.astype(np.float64)
    slack = np.maximum(0.0, -eps - Gt)
    return {"status": str(prob.status), "objective": float(prob.value),
            "d": d_star, "directional_derivative": float(g_D @ d_star.astype(np.float64)),
            "d_norm": float(np.linalg.norm(d_star)),
            "slack_mean": float(slack.mean()), "slack_max": float(slack.max()),
            "violation_max": float(slack.max()), "num_constraints": int(J)}
EOF

############################## core/task_geometry.py ##############################
cat > core/task_geometry.py << 'EOF'
"""Source Constraint Set + Pairwise Margin + phi-gradients."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def select_constraint_samples(model, patches, labels, device,
                              per_class=12, top_competitors=2):
    """Return (patch_index_tensor, true_labels, competitor_labels) for
    correctly-classified source patches, class-balanced.
    """
    model.eval()
    idx = np.arange(len(patches))
    sel_idx, sel_y, sel_k = [], [], []
    with torch.no_grad():
        logits_all = []
        for start in range(0, len(patches), 256):
            batch = torch.from_numpy(patches[start:start+256]).to(device)
            logits_all.append(model(batch).logits.float())
        logits = torch.cat(logits_all, 0).cpu()
    pred = logits.argmax(1)
    for c in range(int(labels.max()) + 1):
        mask = (labels == c) & (pred.numpy() == c)
        ids = idx[mask]
        if len(ids) == 0:
            continue
        margins = []
        for i in ids:
            z = logits[i]
            true_logit = z[c].item()
            other = z.clone(); other[c] = -1e9
            topk = other.topk(top_competitors)
            margins.append(true_logit - topk.values[0].item())
        margins = np.asarray(margins)
        order = np.argsort(margins)
        n = min(per_class, len(ids))
        half = n // 2
        pick_boundary = order[:half]
        pick_easy = order[-half:] if half > 0 else np.array([], dtype=int)
        chosen = np.concatenate([pick_boundary, pick_easy])
        for i_local in chosen:
            gi = ids[int(i_local)]
            z = logits[gi]
            other = z.clone(); other[c] = -1e9
            k = int(other.topk(top_competitors).indices[0])
            sel_idx.append(int(gi)); sel_y.append(int(c)); sel_k.append(k)
    return (torch.as_tensor(sel_idx, dtype=torch.long),
            torch.as_tensor(sel_y, dtype=torch.long),
            torch.as_tensor(sel_k, dtype=torch.long))


def compute_margin_gradients(model, transform, patches, y_true, y_comp, device,
                             batch_size=8):
    """For each sample compute grad_phi (z_y - z_k) at phi=0.
    Returns (G numpy [J,2K], margins0 numpy [J], weights numpy [J]).
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dim = transform.dim
    G_list, m0_list, w_list = [], [], []
    for start in range(0, len(patches), batch_size):
        end = min(start + batch_size, len(patches))
        for i in range(start, end):
            x = torch.from_numpy(patches[i:i+1]).to(device)
            y = int(y_true[i].item()); k = int(y_comp[i].item())
            phi = torch.zeros(dim, device=device, requires_grad=True)
            x_phi = transform(x, phi)
            out = model(x_phi)
            logit = out.logits
            m = logit[0, y] - logit[0, k]
            grad = torch.autograd.grad(m, phi, create_graph=False)[0]
            G_list.append(grad.detach().cpu().numpy())
            m0_list.append(float(m.detach()))
            w_list.append(1.0)
    for p in model.parameters():
        p.requires_grad_(True)
    return (np.stack(G_list, 0).astype(np.float32),
            np.asarray(m0_list, dtype=np.float32),
            np.asarray(w_list, dtype=np.float32))
EOF

############################## core/risk_controller.py ##############################
cat > core/risk_controller.py << 'EOF'
"""Risk-budgeted backtracking line search."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from core.discrepancy import spectral_descriptor, sliced_wasserstein


@torch.no_grad()
def _eval_margins(model, patches, labels, device):
    model.eval()
    logits_batches = []
    for start in range(0, len(patches), 128):
        batch = torch.from_numpy(patches[start:start+128]).to(device)
        logits_batches.append(model(batch).logits.float())
    logits = torch.cat(logits_batches, 0).cpu()
    y = torch.from_numpy(labels).long()
    pred = logits.argmax(1)
    correct = pred.eq(y)
    true_logit = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.clone(); other.scatter_(1, y[:, None], -1e9)
    margins = true_logit - other.max(1).values
    return pred, correct, margins


@torch.no_grad()
def backtracking_search(model, transform, d_star, source_cal_patches,
                        source_cal_labels, target_cal_patches,
                        eta_candidates, tau_sem, tau_flip,
                        tau_class_flip, tau_swd_min_drop, device,
                        swd_seed=0):
    """Return (eta_star, phi_star_tensor, diagnostics_dict)."""
    model.eval()
    d_star_t = torch.from_numpy(d_star).to(device)
    dim = transform.dim

    # baseline (identity) stats
    pred0, correct0, margin0 = _eval_margins(model, source_cal_patches, source_cal_labels, device)
    src_desc0 = spectral_descriptor(torch.from_numpy(source_cal_patches[:64]).to(device))
    tgt_desc = spectral_descriptor(torch.from_numpy(target_cal_patches[:64]).to(device))
    swd0 = sliced_wasserstein(src_desc0, tgt_desc, seed=swd_seed)

    n_classes = int(source_cal_labels.max()) + 1
    cal_class_counts = np.bincount(source_cal_labels, minlength=n_classes).astype(np.float64)

    diagnostics = {"swd_before": float(swd0), "candidates": []}

    for eta in eta_candidates:
        phi = eta * d_star_t
        # forward on cal set with the candidate transform
        logits_batches = []
        for start in range(0, len(source_cal_patches), 128):
            xb = torch.from_numpy(source_cal_patches[start:start+128]).to(device)
            xb_t = transform(xb, phi)
            logits_batches.append(model(xb_t).logits.float())
        logits1 = torch.cat(logits_batches, 0).cpu()
        pred1 = logits1.argmax(1)
        correct1 = pred1.eq(torch.from_numpy(source_cal_labels).long())

        # new error rate
        new_err = (correct0 & (~correct1))
        V = float(new_err.float().mean().item())

        # class-wise
        Vc = np.zeros(n_classes, dtype=np.float64)
        for c in range(n_classes):
            mask = (torch.from_numpy(source_cal_labels).long() == c)
            if cal_class_counts[c] > 0:
                Vc[c] = float((new_err & mask).float().sum().item()) / max(cal_class_counts[c], 1.0)
        Vmax = float(Vc.max()) if Vc.size else 0.0

        # margin decay
        y = torch.from_numpy(source_cal_labels).long()
        true_logit1 = logits1.gather(1, y[:, None]).squeeze(1)
        other1 = logits1.clone(); other1.scatter_(1, y[:, None], -1e9)
        margin1 = true_logit1 - other1.max(1).values
        delta = (margin0 - margin1)
        Rsem = float((torch.relu(delta - 0.05) ** 2).mean().item())

        # SWD improvement
        x_cal_t = transform(torch.from_numpy(source_cal_patches[:64]).to(device), phi)
        src_desc1 = spectral_descriptor(x_cal_t)
        swd1 = sliced_wasserstein(src_desc1, tgt_desc, seed=swd_seed)
        swd_drop = float(swd0 - swd1)

        diag = {
            "eta": float(eta), "V": V, "Vmax": Vmax, "Rsem": Rsem,
            "swd_before": float(swd0), "swd_after": float(swd1),
            "swd_drop": swd_drop,
            "spectral": transform.diagnostics(
                torch.from_numpy(source_cal_patches[:32]).to(device), phi),
        }
        diagnostics["candidates"].append(diag)

        if (V <= tau_flip and Vmax <= tau_class_flip
                and Rsem <= tau_sem and swd_drop >= tau_swd_min_drop):
            return float(eta), phi, diagnostics

    return 0.0, torch.zeros(dim, device=device), diagnostics
EOF

############################## core/metrics.py ##############################
cat > core/metrics.py << 'EOF'
"""Classification metrics."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes):
    m = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            m[t, p] += 1
    return m


def classification_metrics(m):
    total = int(m.sum())
    correct = int(np.trace(m))
    oa = 100.0 * correct / max(1, total)
    row = m.sum(1)
    class_acc = np.divide(np.diag(m), row, out=np.zeros_like(row, dtype=np.float64),
                          where=row > 0)
    aa = 100.0 * float(class_acc.mean())
    col = m.sum(0)
    expected = float(np.dot(row, col)) / max(1.0, float(total * total))
    obs = float(correct) / max(1.0, float(total))
    kappa = (obs - expected) / max(1e-12, 1.0 - expected)

    # macro F1
    f1s = []
    for c in range(m.shape[0]):
        tp = m[c, c]
        fp = m[:, c].sum() - tp
        fn = m[c, :].sum() - tp
        if tp + fp + fn == 0:
            continue
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    macro_f1 = 100.0 * float(np.mean(f1s)) if f1s else 0.0

    return {"OA": oa, "AA": aa, "Kappa": float(kappa), "MacroF1": macro_f1,
            "class_accuracy": (100.0 * class_acc).tolist(),
            "confusion_matrix": m.tolist()}
EOF

############################## core/data.py ##############################
cat > core/data.py << 'EOF'
"""Auto-detecting .mat loader + patch extraction."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


IMAGE_KEYS = ("ori_data", "data", "image", "img", "HSI", "hsi", "X", "cube")
GT_KEYS = ("map", "gt", "GT", "ground_truth", "label", "labels", "Y", "gt_map")


def _load_mat(path):
    try:
        return {k: v for k, v in sio.loadmat(str(path)).items() if not k.startswith("__")}
    except (NotImplementedError, ValueError):
        with h5py.File(path, "r") as f:
            return {k: np.asarray(f[k][:]) for k in f.keys()}


def _pick_3d(d):
    for k in IMAGE_KEYS:
        if k in d and isinstance(d[k], np.ndarray) and d[k].ndim == 3:
            return d[k], k
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            return v, k
    return None, None


def _pick_2d(d):
    for k in GT_KEYS:
        if k in d and isinstance(d[k], np.ndarray) and d[k].ndim == 2:
            return d[k], k
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            return v, k
    return None, None


def _band_axis(arr, expected_bands):
    """Find the axis that most likely is the spectral band axis."""
    if expected_bands is not None and expected_bands in arr.shape:
        return int(arr.shape.index(expected_bands))
    # heuristic: axis with size in [20, 500]
    candidates = [i for i, s in enumerate(arr.shape) if 20 <= s <= 500]
    if len(candidates) == 1:
        return candidates[0]
    # fallback: smallest axis
    return int(np.argmin(arr.shape))


def load_scene(image_path, gt_path, expected_bands, num_classes, verbose=True):
    image_path = Path(image_path)
    d = _load_mat(image_path)
    cube, key = _pick_3d(d)
    if cube is None:
        raise KeyError(f"{image_path}: no 3-D array found. keys={list(d.keys())}")
    b_axis = _band_axis(cube, expected_bands)
    if b_axis != 2:
        cube = np.moveaxis(cube, b_axis, 2)
    cube = np.asarray(cube, dtype=np.float32)
    if verbose:
        print(f"[load] {image_path.name} key={key} raw_shape={cube.shape} "
              f"band_axis_from={b_axis} num_classes={num_classes}")

    gt = None
    if gt_path is not None:
        gd = _load_mat(Path(gt_path))
        gt_arr, gkey = _pick_2d(gd)
        if gt_arr is None:
            raise KeyError(f"{gt_path}: no 2-D array found. keys={list(gd.keys())}")
        gt = np.asarray(gt_arr, dtype=np.int64)
        if gt.shape != cube.shape[:2]:
            raise ValueError(f"image {cube.shape[:2]} vs gt {gt.shape}")
        if verbose:
            print(f"[load] {Path(gt_path).name} key={gkey} shape={gt.shape} "
                  f"classes={sorted(np.unique(gt).tolist())}")

    finite = np.isfinite(cube).all(-1)
    energy = np.linalg.norm(np.nan_to_num(cube, copy=False), axis=-1)
    valid = finite & (energy > 0)
    cube[~np.isfinite(cube)] = 0.0
    scale = float(np.abs(cube[valid]).max()) if valid.any() else 1.0
    cube = cube / max(scale, 1e-6)
    norms = np.linalg.norm(cube, axis=-1, keepdims=True)
    cube = cube / np.maximum(norms, 1e-6)
    cube[~valid] = 0.0
    return cube, gt


def align_bands(src_cube, tgt_cube, src_stride=1):
    """Match band counts by optional stride on source then uniform on smaller."""
    if src_stride > 1:
        src_cube = src_cube[:, :, ::src_stride]
    bs, bt = src_cube.shape[2], tgt_cube.shape[2]
    target_bands = min(bs, bt)
    if bs != target_bands:
        idx = np.linspace(0, bs - 1, target_bands).round().astype(int)
        src_cube = src_cube[:, :, idx]
    if bt != target_bands:
        idx = np.linspace(0, bt - 1, target_bands).round().astype(int)
        tgt_cube = tgt_cube[:, :, idx]
    return src_cube, tgt_cube, target_bands


def build_patches(image, coords, patch_size):
    r = patch_size // 2
    pad = ((r, r), (r, r), (0, 0))
    img = np.pad(image, pad, mode="reflect")
    patches = np.empty((len(coords), image.shape[2], patch_size, patch_size),
                       dtype=np.float32)
    for i, (row, col) in enumerate(coords):
        rr, cc = row + r, col + r
        patches[i] = img[rr-r:rr+r+1, cc-r:cc+r+1].transpose(2, 0, 1)
    return patches


def stratified_split(gt, ratios, seed):
    """ratios: dict with keys 'train','constraint','calibration','test'."""
    rng = np.random.default_rng(seed)
    masks = {k: np.zeros_like(gt, dtype=bool) for k in ratios}
    for c in [int(i) for i in np.unique(gt) if i > 0]:
        coords = np.argwhere(gt == c)
        rng.shuffle(coords)
        n = len(coords)
        n_tr = max(1, int(round(n * ratios["train"])))
        n_cn = max(1, int(round(n * ratios["constraint"])))
        n_ca = max(1, int(round(n * ratios["calibration"])))
        n_te = n - n_tr - n_cn - n_ca
        if n_te < 0:
            n_te = 0
        slices = {"train": coords[:n_tr],
                  "constraint": coords[n_tr:n_tr+n_cn],
                  "calibration": coords[n_tr+n_cn:n_tr+n_cn+n_ca],
                  "test": coords[n_tr+n_cn+n_ca:n_tr+n_cn+n_ca+n_te]}
        for k, arr in slices.items():
            if len(arr):
                masks[k][arr[:, 0], arr[:, 1]] = True
    return masks


def sample_unlabeled(image, target_mask, n, seed):
    valid = np.argwhere(target_mask)
    rng = np.random.default_rng(seed)
    if len(valid) > n:
        idx = rng.choice(len(valid), size=n, replace=False)
        valid = valid[idx]
    return valid
EOF

############################## core/trainer_v8.py ##############################
cat > core/trainer_v8.py << 'EOF'
"""T-CST v8.0 training loop: warm-up + alternating outer cycles."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from core.data import (align_bands, build_patches, load_scene,
                       sample_unlabeled, stratified_split)
from core.discrepancy import (domain_gradient_mmd, spectral_descriptor,
                              sliced_wasserstein)
from core.metrics import classification_metrics, confusion_matrix
from core.model import DualReliabilityEncoder
from core.risk_controller import backtracking_search
from core.spectral_transform import SmoothSpectralTransform
from core.task_geometry import (compute_margin_gradients,
                                select_constraint_samples)
from core.transport_solver import solve_margin_qp


def _make_model(bands, num_classes, device):
    m = DualReliabilityEncoder(
        bands=bands, num_classes=num_classes,
        projection_dim=128, projection_norm="batch",
        aspp_mode="adaptive", pooling_mode="guided",
        spectral_fusion_mode="reliable", amplitude_channel=False)
    return m.to(device)


def _eval_source(model, patches, labels, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(patches), 256):
            xb = torch.from_numpy(patches[start:start+256]).to(device)
            yb = torch.from_numpy(labels[start:start+256]).to(device)
            pred = model(xb).logits.argmax(1)
            correct += int(pred.eq(yb).sum().item())
    return 100.0 * correct / max(1, len(patches))


def _eval_target(model, patches, labels, device, num_classes):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(patches), 256):
            xb = torch.from_numpy(patches[start:start+256]).to(device)
            preds.append(model(xb).logits.argmax(1).cpu().numpy())
    pred = np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)
    m = confusion_matrix(labels, pred, num_classes)
    return classification_metrics(m)


def run_tcst_v8(cfg, device):
    torch.manual_seed(cfg["output"]["seed"])
    np.random.seed(cfg["output"]["seed"])

    # ------------------------- load data -------------------------
    print(f"\n{'='*70}\nDataset: {cfg['dataset']['name']}\n{'='*70}")
    src_cube, src_gt = load_scene(cfg["dataset"]["source_image"],
                                  cfg["dataset"]["source_gt"],
                                  cfg["dataset"]["expected_bands"]["source"],
                                  cfg["dataset"]["num_classes"], verbose=True)
    tgt_cube, tgt_gt = load_scene(cfg["dataset"]["target_image"],
                                  cfg["dataset"]["target_gt"],
                                  cfg["dataset"]["expected_bands"]["target"],
                                  cfg["dataset"]["num_classes"], verbose=True)
    src_cube, tgt_cube, bands = align_bands(src_cube, tgt_cube,
                                             cfg["dataset"].get("band_stride_source", 1))
    print(f"[align] bands={bands} num_classes={cfg['dataset']['num_classes']}")

    num_classes = cfg["dataset"]["num_classes"]
    patch_size = cfg["training"]["patch_size"]

    # source split: train / constraint / calibration (from source GT)
    split = stratified_split(src_gt, {"train": 0.8, "constraint": 0.10,
                                      "calibration": 0.10, "test": 0.0},
                             seed=cfg["output"]["seed"] + 11)
    def _coords_labels(mask):
        coords = np.argwhere(mask)
        labels = src_gt[coords[:, 0], coords[:, 1]] - 1
        return coords, labels.astype(np.int64)

    src_tr_c, src_tr_y = _coords_labels(split["train"])
    src_cn_c, src_cn_y = _coords_labels(split["constraint"])
    src_ca_c, src_ca_y = _coords_labels(split["calibration"])
    print(f"[source-split] train={len(src_tr_c)} constraint={len(src_cn_c)} "
          f"calibration={len(src_ca_c)}")

    src_tr_p = build_patches(src_cube, src_tr_c, patch_size)
    src_cn_p = build_patches(src_cube, src_cn_c, patch_size)
    src_ca_p = build_patches(src_cube, src_ca_c, patch_size)

    # target pool (no GT used during training)
    tgt_valid_mask = np.linalg.norm(tgt_cube, axis=-1) > 0
    tgt_pool_c = sample_unlabeled(tgt_cube, tgt_valid_mask,
                                  cfg["training"].get("target_pool_size", 4000),
                                  seed=cfg["output"]["seed"] + 17)
    tgt_pool_p = build_patches(tgt_cube, tgt_pool_c, patch_size)
    print(f"[target] unlabeled pool = {len(tgt_pool_c)}")

    # target eval set (GT only for the final metric)
    if tgt_gt is not None:
        tgt_eval_c = np.argwhere((tgt_gt > 0) &
                                 (np.linalg.norm(tgt_cube, axis=-1) > 0))
        tgt_eval_y = (tgt_gt[tgt_eval_c[:, 0], tgt_eval_c[:, 1]] - 1).astype(np.int64)
        tgt_eval_p = build_patches(tgt_cube, tgt_eval_c, patch_size)
        print(f"[target-eval] usable GT pixels = {len(tgt_eval_c)}")
    else:
        tgt_eval_p = tgt_eval_y = None

    # ------------------------- model + transform -------------------------
    model = _make_model(bands, num_classes, device)
    transform = SmoothSpectralTransform(bands,
                                        K=cfg["tcst"]["K"],
                                        rho_a=cfg["tcst"]["rho_a"],
                                        rho_b=cfg["tcst"]["rho_b"]).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"],
                      weight_decay=cfg["training"]["weight_decay"])
    amp = cfg["training"]["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch, "amp") else \
             torch.cuda.amp.GradScaler(enabled=amp)

    save_dir = Path(cfg["output"]["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "training_log.jsonl"

    def _train_step(patches, labels, phi_star, safe_weight):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        xb = torch.from_numpy(patches).to(device)
        yb = torch.from_numpy(labels).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits_clean = model(xb).logits
            loss_clean = F.cross_entropy(logits_clean, yb, label_smoothing=0.035)
            if phi_star is not None and float(phi_star.abs().sum()) > 1e-8:
                xb_safe = transform(xb, phi_star)
                logits_safe = model(xb_safe).logits
                loss_safe = F.cross_entropy(logits_safe, yb, label_smoothing=0.035)
                loss = (loss_clean + safe_weight * loss_safe) / (1 + safe_weight)
            else:
                loss = loss_clean
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        return float(loss.detach())

    # ------------------------- Phase A: warm-up -------------------------
    print(f"\n[Phase A] warm-up {cfg['training']['epochs_warmup']} epochs")
    warmup = cfg["training"]["epochs_warmup"]
    batch = cfg["training"]["batch_size"]
    rng = np.random.default_rng(cfg["output"]["seed"] + 21)
    best_src = 0.0
    best_state = None
    t0 = time.time()
    for ep in range(1, warmup + 1):
        idx = rng.permutation(len(src_tr_p))
        losses = []
        for start in range(0, len(idx), batch):
            sel = idx[start:start+batch]
            loss = _train_step(src_tr_p[sel], src_tr_y[sel], None, 0.0)
            losses.append(loss)
        src_acc = _eval_source(model, src_ca_p, src_ca_y, device)
        if src_acc > best_src:
            best_src = src_acc
            best_state = copy.deepcopy(model.state_dict())
        print(f"  warm {ep:03d}/{warmup}  loss={np.mean(losses):.4f}  "
              f"src_cal_OA={src_acc:.2f}  best={best_src:.2f}  "
              f"t={(time.time()-t0)/60:.1f}m")
    if best_state is not None:
        model.load_state_dict(best_state)

    # ------------------------- Phase B+C: alternating cycles -------------------------
    print(f"\n[Phase B+C] {cfg['training']['num_cycles']} cycles × "
          f"{cfg['training']['epochs_per_cycle']} epochs")
    cycle_logs = []
    for cycle in range(1, cfg["training"]["num_cycles"] + 1):
        print(f"\n--- Cycle {cycle} ---")
        model.eval()
        # ---------- transport estimation (Phase B) ----------
        # target descriptor uses the unlabeled pool
        tgt_desc = spectral_descriptor(torch.from_numpy(tgt_pool_p[:128]).to(device))
        src_tr_for_mmd = torch.from_numpy(src_tr_p[:128]).to(device)
        g_D, mmd0 = domain_gradient_mmd(transform, src_tr_for_mmd, tgt_desc)
        g_D_np = g_D.cpu().numpy().astype(np.float32)
        print(f"  [transport] MMD^2(phi=0)={mmd0:.4f}  ||g_D||={np.linalg.norm(g_D_np):.4f}")

        # task geometry
        sel_idx, sel_y, sel_k = select_constraint_samples(
            model, src_cn_p, src_cn_y, device,
            per_class=cfg["tcst"]["per_class_constraint_samples"],
            top_competitors=cfg["tcst"].get("top_competitors", 2))
        if len(sel_idx) == 0:
            print("  [transport] no correct source samples for constraints; skip")
            phi_star = torch.zeros(transform.dim, device=device)
            diag = {"skipped": "no_constraint_samples"}
        else:
            G_np, m0_np, w_np = compute_margin_gradients(
                model, transform, src_cn_p[sel_idx.numpy()],
                sel_y, sel_k, device)
            print(f"  [transport] margin constraints: J={G_np.shape[0]}  "
                  f"mean_m0={m0_np.mean():.3f}")

            qp = solve_margin_qp(g_D_np, G_np, m0_np, w_np,
                                 rho_m=cfg["tcst"]["rho_m"],
                                 lambda_d=cfg["tcst"]["lambda_d"],
                                 lambda_xi=cfg["tcst"]["lambda_xi"])
            print(f"  [qp] status={qp['status']}  "
                  f"||d*||={qp['d_norm']:.4f}  <g_D,d*>={qp['directional_derivative']:.4f}  "
                  f"slack_max={qp['slack_max']:.4f}")

            if qp["status"] not in ("optimal", "no_constraints") or qp["d_norm"] < 1e-8:
                phi_star = torch.zeros(transform.dim, device=device)
                diag = {"qp_status": qp["status"], "d_norm": qp["d_norm"]}
            else:
                eta_star, phi_star, diag = backtracking_search(
                    model, transform, qp["d"],
                    src_ca_p, src_ca_y,
                    tgt_pool_p[:64],
                    eta_candidates=cfg["tcst"]["eta_candidates"],
                    tau_sem=cfg["tcst"]["tau_sem"],
                    tau_flip=cfg["tcst"]["tau_flip"],
                    tau_class_flip=cfg["tcst"]["tau_class_flip"],
                    tau_swd_min_drop=cfg["tcst"]["tau_swd_min_drop"],
                    device=device, swd_seed=cfg["output"]["seed"])
                print(f"  [risk] accepted eta={eta_star:.4f}")

        # ---------- safe training (Phase C) ----------
        rng = np.random.default_rng(cfg["output"]["seed"] + 100 + cycle)
        for ep in range(1, cfg["training"]["epochs_per_cycle"] + 1):
            idx = rng.permutation(len(src_tr_p))
            losses = []
            for start in range(0, len(idx), batch):
                sel = idx[start:start+batch]
                loss = _train_step(src_tr_p[sel], src_tr_y[sel],
                                   phi_star, cfg["tcst"]["lambda_safe"])
                losses.append(loss)
            src_acc = _eval_source(model, src_ca_p, src_ca_y, device)
            print(f"    cyc{cycle} ep{ep:02d}  loss={np.mean(losses):.4f}  "
                  f"src_cal_OA={src_acc:.2f}")

        cycle_logs.append({"cycle": cycle, "diag": diag})
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"stage": "cycle", "cycle": cycle,
                                "transport": diag}, default=str) + "\n")

    # ------------------------- final evaluation -------------------------
    print(f"\n[Final evaluation] using target GT (never used during training)")
    result = {
        "dataset": cfg["dataset"]["name"],
        "bands": int(bands),
        "num_classes": int(num_classes),
        "cycles": cycle_logs,
    }
    if tgt_eval_p is not None:
        metrics = _eval_target(model, tgt_eval_p, tgt_eval_y, device, num_classes)
        result["target_metrics"] = metrics
        print(f"\n{'='*70}")
        print(f"TARGET RESULT — {cfg['dataset']['name']}")
        print(f"{'='*70}")
        print(f"OA     = {metrics['OA']:.2f}")
        print(f"AA     = {metrics['AA']:.2f}")
        print(f"Kappa  = {metrics['Kappa']:.4f}")
        print(f"MacroF1= {metrics['MacroF1']:.2f}")
        print(f"Per-class accuracy: {[round(x,2) for x in metrics['class_accuracy']]}")
        print(f"{'='*70}")
    else:
        print("  (no target GT available — skipping final metric)")

    # save artifacts
    checkpoint = save_dir / "final_state.pt"
    torch.save({"model": model.state_dict(),
                "transform": transform.state_dict(),
                "args": cfg, "metrics": result.get("target_metrics")}, checkpoint)
    (save_dir / "final_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {checkpoint}")
    print(f"Saved: {save_dir / 'final_metrics.json'}")
    return result
EOF

############################## configs/houston.yaml ##############################
cat > configs/houston.yaml << 'EOF'
dataset:
  name: houston
  source_name: Houston13
  target_name: Houston18
  data_root: /root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/Houston
  source_image: /root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/Houston/Houston13.mat
  source_gt:    /root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/Houston/Houston13_7gt.mat
  target_image: /root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/Houston/Houston18.mat
  target_gt:    /root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/Houston/Houston18_7gt.mat
  num_classes: 7
  expected_bands:
    source: 48
    target: 48
  band_stride_source: 3   # 如果 source 已经是 48 波段，改成 1

training:
  patch_size: 11
  batch_size: 64
  epochs_warmup: 15
  epochs_per_cycle: 5
  num_cycles: 3
  lr: 0.0003
  weight_decay: 0.0004
  num_workers: 4
  amp: true
  target_pool_size: 4000

tcst:
  K: 8
  rho_a: 0.10
  rho_b: 0.10
  rho_m: 0.05
  lambda_d: 1.0
  lambda_xi: 10.0
  lambda_safe: 1.0
  per_class_constraint_samples: 12
  top_competitors: 2
  eta_candidates: [1.0, 0.5, 0.25, 0.125, 0.0625]
  tau_sem: 0.05
  tau_flip: 0.02
  tau_class_flip: 0.10
  tau_swd_min_drop: 1e-3

output:
  save_dir: /root/autodl-tmp/TCSTdemo/results/houston
  seed: 1237
EOF

############################## configs/hyrank.yaml ##############################
cat > configs/hyrank.yaml << 'EOF'
dataset:
  name: hyrank
  source_name: Dioni
  target_name: Loukia
  data_root: /root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/HyRANK
  source_image: /root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/HyRANK/Dioni.mat
  source_gt:    /root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/HyRANK/Dioni_gt.mat
  target_image: /root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/HyRANK/Loukia.mat
  target_gt:    /root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/HyRANK/Loukia_gt.mat
  num_classes: 12
  expected_bands:
    source: 176
    target: 176
  band_stride_source: 1

training:
  patch_size: 11
  batch_size: 64
  epochs_warmup: 15
  epochs_per_cycle: 5
  num_cycles: 3
  lr: 0.0003
  weight_decay: 0.0004
  num_workers: 4
  amp: true
  target_pool_size: 4000

tcst:
  K: 8
  rho_a: 0.10
  rho_b: 0.10
  rho_m: 0.05
  lambda_d: 1.0
  lambda_xi: 10.0
  lambda_safe: 1.0
  per_class_constraint_samples: 12
  top_competitors: 2
  eta_candidates: [1.0, 0.5, 0.25, 0.125, 0.0625]
  tau_sem: 0.05
  tau_flip: 0.02
  tau_class_flip: 0.10
  tau_swd_min_drop: 1e-3

output:
  save_dir: /root/autodl-tmp/TCSTdemo/results/hyrank
  seed: 1237
EOF

############################## configs/shanghai.yaml ##############################
cat > configs/shanghai.yaml << 'EOF'
dataset:
  name: shanghai
  source_name: Shanghai
  target_name: Hangzhou
  data_root: /root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/Shanghai_Hangzhou
  source_image: /root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/Shanghai_Hangzhou/Shanghai.mat
  source_gt:    /root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/Shanghai_Hangzhou/Shanghai_gt.mat
  target_image: /root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/Shanghai_Hangzhou/Hangzhou.mat
  target_gt:    /root/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets/Shanghai_Hangzhou/Hangzhou_gt.mat
  num_classes: 3
  expected_bands:
    source: 198
    target: 198
  band_stride_source: 1

training:
  patch_size: 11
  batch_size: 64
  epochs_warmup: 15
  epochs_per_cycle: 5
  num_cycles: 3
  lr: 0.0003
  weight_decay: 0.0004
  num_workers: 4
  amp: true
  target_pool_size: 4000

tcst:
  K: 8
  rho_a: 0.10
  rho_b: 0.10
  rho_m: 0.05
  lambda_d: 1.0
  lambda_xi: 10.0
  lambda_safe: 1.0
  per_class_constraint_samples: 12
  top_competitors: 2
  eta_candidates: [1.0, 0.5, 0.25, 0.125, 0.0625]
  tau_sem: 0.05
  tau_flip: 0.02
  tau_class_flip: 0.10
  tau_swd_min_drop: 1e-3

output:
  save_dir: /root/autodl-tmp/TCSTdemo/results/shanghai
  seed: 1237
EOF

############################## scripts/train.py ##############################
cat > scripts/train.py << 'EOF'
#!/usr/bin/env python3
"""Universal entry: python scripts/train.py --config configs/houston.yaml"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from core.trainer_v8 import run_tcst_v8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(args.device if args.device
                          else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[device] {device}")
    run_tcst_v8(cfg, device)


if __name__ == "__main__":
    main()
EOF

############################## scripts/print_results.py ##############################
cat > scripts/print_results.py << 'EOF'
#!/usr/bin/env python3
"""Print a full report from a results directory or json file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def show(path: Path):
    d = json.loads(path.read_text())
    print(f"\n{'='*70}\n{path}\n{'='*70}")
    print(f"Dataset: {d.get('dataset')}   bands={d.get('bands')}   "
          f"num_classes={d.get('num_classes')}")
    cycles = d.get("cycles", [])
    for c in cycles:
        tr = c.get("diag", {})
        if "qp_status" in tr:
            print(f"  Cycle {c['cycle']}: QP={tr['qp_status']}  "
                  f"||d*||={tr.get('d_norm', 'NA')}")
        if "candidates" in tr and tr["candidates"]:
            last = tr["candidates"][-1]
            print(f"  Cycle {c['cycle']}: last_cand eta={last['eta']:.3f} "
                  f"V={last['V']:.4f} Rsem={last['Rsem']:.4f} "
                  f"SWDdrop={last['swd_drop']:.4f}")
    m = d.get("target_metrics")
    if m:
        print(f"\nTarget OA={m['OA']:.2f}  AA={m['AA']:.2f}  "
              f"Kappa={m['Kappa']:.4f}  MacroF1={m['MacroF1']:.2f}")
        print(f"Per-class: {[round(x,2) for x in m['class_accuracy']]}")
        cm = m.get("confusion_matrix", [])
        if cm:
            print("\nConfusion matrix:")
            for row in cm:
                print("  " + " ".join(f"{v:4d}" for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True,
                    help="results directory or a specific final_metrics.json")
    args = ap.parse_args()
    p = Path(args.path)
    if p.is_dir():
        files = sorted(p.rglob("final_metrics.json"))
        if not files:
            print(f"No final_metrics.json under {p}")
            return
        for f in files:
            show(f)
    else:
        show(p)


if __name__ == "__main__":
    main()
EOF

############################## tests/test_core.py ##############################
cat > tests/test_core.py << 'EOF'
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from core.spectral_transform import SmoothSpectralTransform, build_dct_basis
from core.discrepancy import (spectral_descriptor, mk_mmd2,
                              domain_gradient_mmd, sliced_wasserstein)
from core.transport_solver import solve_margin_qp
from core.model import DualReliabilityEncoder


def test_identity():
    T = SmoothSpectralTransform(48, K=8)
    x = torch.randn(4, 48, 9, 9)
    y = T(x, torch.zeros(16))
    assert torch.allclose(x, y, atol=1e-6)


def test_dct():
    U = build_dct_basis(48, 8)
    assert torch.allclose(U.t() @ U, torch.eye(8), atol=1e-4)


def test_descriptor():
    d = spectral_descriptor(torch.randn(5, 48, 9, 9))
    assert d.shape == (5, 4*48-3)


def test_grad():
    T = SmoothSpectralTransform(48, K=8)
    g, m = domain_gradient_mmd(T, torch.randn(8, 48, 9, 9),
                               spectral_descriptor(torch.randn(8, 48, 9, 9)))
    assert g.shape == (16,) and torch.isfinite(g).all() and m >= 0


def test_swd_zero():
    x = torch.randn(8, 4)
    assert abs(sliced_wasserstein(x, x, seed=0)) < 1e-6


def test_qp():
    g_D = np.array([1.0, 1.0], dtype=np.float32)
    out = solve_margin_qp(g_D, np.zeros((0, 2), np.float32),
                          np.zeros((0,), np.float32), np.zeros((0,), np.float32))
    assert out["directional_derivative"] <= 0


def test_backbone_forward():
    m = DualReliabilityEncoder(bands=48, num_classes=7,
                               projection_dim=64, projection_norm="batch")
    out = m(torch.randn(2, 48, 11, 11))
    assert out.logits.shape == (2, 7)


if __name__ == "__main__":
    test_identity(); test_dct(); test_descriptor()
    test_grad(); test_swd_zero(); test_qp(); test_backbone_forward()
    print("ALL TESTS PASSED")
EOF

############################## README.md ##############################
cat > README.md << 'EOF'
# T-CST v8.0 Core
Quick-start:
  pip install -r requirements.txt
  python tests/test_core.py
  python scripts/train.py --config configs/houston.yaml
  python scripts/print_results.py --path results/houston
EOF

chmod +x scripts/train.py scripts/print_results.py
echo "SETUP COMPLETE"
