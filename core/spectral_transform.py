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


def class_residual_reliability(probabilities, min_effective_samples=16,
                               full_effective_samples=128, min_confidence=0.75):
    probabilities = np.asarray(probabilities, dtype=np.float64).copy()
    if (probabilities.ndim != 2 or probabilities.shape[0] == 0
            or probabilities.shape[1] < 2):
        raise ValueError("probabilities must be a non-empty [samples, classes] matrix")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities.sum(axis=1) <= 0).any()):
        raise ValueError("probabilities must be finite, non-negative, and have positive row sums")
    if (min_effective_samples < 0 or full_effective_samples <= min_effective_samples
            or not 0 <= min_confidence < 1):
        raise ValueError("invalid class reliability thresholds")

    probabilities /= probabilities.sum(axis=1, keepdims=True)
    mass = probabilities.sum(axis=0)
    squared_mass = (probabilities ** 2).sum(axis=0)
    effective = mass ** 2 / np.maximum(squared_mass, 1e-12)
    confidence = squared_mass / np.maximum(mass, 1e-12)
    support_gate = np.clip(
        (effective - min_effective_samples) /
        (full_effective_samples - min_effective_samples), 0.0, 1.0)
    confidence_gate = np.clip(
        (confidence - min_confidence) / (1.0 - min_confidence), 0.0, 1.0)
    return (support_gate * confidence_gate).astype(np.float32)


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

    def forward(self, x, phi, labels=None, class_probs=None):
        # labels/class_probs are ignored for global transform (backward compat)
        if x.shape[1] != self.bands:
            raise ValueError(f"Expected {self.bands} bands, got {x.shape[1]}")
        a = self.gain(phi[: self.K]).view(1, -1, 1, 1)
        b = self.offset(phi[self.K:]).view(1, -1, 1, 1)
        return x * a + b

    @torch.no_grad()
    def diagnostics(self, x, phi, labels=None):
        xt = self.forward(x, phi)
        diff = xt - x
        rel = float(diff.norm() / x.norm().clamp_min(1e-8))
        a = self.gain(phi[: self.K])
        b = self.offset(phi[self.K:])
        # 总变差（TV）：相邻波段增益/偏移的绝对差均值，用来捕捉剧烈跳变
        gain_tv = float((a[1:] - a[:-1]).abs().mean()) if a.numel() > 1 else 0.0
        offset_tv = float((b[1:] - b[:-1]).abs().mean()) if b.numel() > 1 else 0.0
        return {"relative_spectral_change": rel,
                "gain_min": float(a.min()), "gain_max": float(a.max()),
                "offset_min": float(b.min()), "offset_max": float(b.max()),
                "gain_tv": gain_tv, "offset_tv": offset_tv}


class ClassSmoothSpectralTransform(nn.Module):
    """Per-class smooth spectral transform: each class c has its own (a_c, b_c).

    Source domain: labels=c gives hard assignment (a_c, b_c).
    Target domain: class_probs (N, C) gives soft mixture (sum_c p_c * a_c).
    phi is [C*2K] flat: alpha[C,K] then beta[C,K].
    """

    def __init__(self, bands, num_classes, K=8, rho_a=0.10, rho_b=0.10):
        super().__init__()
        self.bands = int(bands)
        self.num_classes = int(num_classes)
        self.K = int(K)
        self.rho_a = float(rho_a)
        self.rho_b = float(rho_b)
        self.register_buffer("U", build_dct_basis(self.bands, self.K), persistent=True)

    @property
    def dim(self):
        return self.num_classes * 2 * self.K

    def gain_all(self, alpha):
        # alpha: [C, K]; U: [B, K];  -> [C, B]
        return torch.exp(self.rho_a * torch.tanh(self.U @ alpha.t()).t())

    def offset_all(self, beta):
        return self.rho_b * torch.tanh(self.U @ beta.t()).t()

    def _split(self, phi):
        C, K = self.num_classes, self.K
        alpha = phi[: C * K].view(C, K)
        beta = phi[C * K:].view(C, K)
        return alpha, beta

    def forward(self, x, phi, labels=None, class_probs=None):
        if x.shape[1] != self.bands:
            raise ValueError(f"Expected {self.bands} bands, got {x.shape[1]}")
        alpha, beta = self._split(phi)
        a = self.gain_all(alpha)   # [C, B]
        b = self.offset_all(beta)  # [C, B]
        if labels is not None:
            labels = labels.long()
            a_n = a[labels]          # [N, B]
            b_n = b[labels]
        elif class_probs is not None:
            a_n = class_probs @ a    # [N, B]
            b_n = class_probs @ b
        else:
            a_n = a.mean(0, keepdim=True).expand(x.size(0), -1)
            b_n = b.mean(0, keepdim=True).expand(x.size(0), -1)
        a_n = a_n.view(-1, self.bands, 1, 1)
        b_n = b_n.view(-1, self.bands, 1, 1)
        return x * a_n + b_n

    @torch.no_grad()
    def diagnostics(self, x, phi, labels=None):
        xt = self.forward(x, phi, labels=labels)
        diff = xt - x
        rel = float(diff.norm() / x.norm().clamp_min(1e-8))
        alpha, beta = self._split(phi)
        a = self.gain_all(alpha)   # [C, B]
        b = self.offset_all(beta)
        if a.size(1) > 1:
            gain_tv_per = (a[:, 1:] - a[:, :-1]).abs().mean(dim=1)
            offset_tv_per = (b[:, 1:] - b[:, :-1]).abs().mean(dim=1)
            gain_tv = float(gain_tv_per.max())
            offset_tv = float(offset_tv_per.max())
        else:
            gain_tv = offset_tv = 0.0
        return {
            "relative_spectral_change": rel,
            "gain_min": float(a.min()), "gain_max": float(a.max()),
            "offset_min": float(b.min()), "offset_max": float(b.max()),
            "gain_tv": gain_tv, "offset_tv": offset_tv,
        }


class HierarchicalSpectralTransform(nn.Module):
    """Global DCT transform with reliability-gated low-frequency class residuals."""

    def __init__(self, bands, num_classes, K=8, residual_k=2,
                 rho_a=0.10, rho_b=0.10):
        super().__init__()
        if not 0 < residual_k <= K:
            raise ValueError("residual_k must be in (0, K]")
        self.bands = int(bands)
        self.num_classes = int(num_classes)
        self.K = int(K)
        self.residual_k = int(residual_k)
        self.rho_a = float(rho_a)
        self.rho_b = float(rho_b)
        self.register_buffer("U", build_dct_basis(self.bands, self.K), persistent=True)
        self.register_buffer("class_reliability", torch.zeros(self.num_classes), persistent=False)

    @property
    def global_dim(self):
        return 2 * self.K

    @property
    def dim(self):
        return self.global_dim + 2 * self.num_classes * self.residual_k

    @torch.no_grad()
    def set_class_reliability(self, probabilities):
        if torch.is_tensor(probabilities):
            probabilities = probabilities.detach().float().cpu().numpy()
        probabilities = np.asarray(probabilities)
        if probabilities.ndim != 2 or probabilities.shape[1] != self.num_classes:
            raise ValueError(f"Expected probabilities with {self.num_classes} classes")
        reliability = class_residual_reliability(probabilities)
        self.class_reliability.copy_(torch.as_tensor(
            reliability, dtype=self.U.dtype, device=self.U.device))
        return reliability

    def _class_coefficients(self, phi):
        c, r = self.num_classes, self.residual_k
        alpha = phi[self.global_dim:self.global_dim + c * r].view(c, r)
        beta = phi[self.global_dim + c * r:].view(c, r)
        base_alpha = self.U @ phi[:self.K]
        base_beta = self.U @ phi[self.K:self.global_dim]
        residual_alpha = self.U[:, :r] @ alpha.t()
        residual_beta = self.U[:, :r] @ beta.t()
        gates = self.class_reliability.to(phi.device)
        gains = torch.exp(self.rho_a * torch.tanh(
            base_alpha[None, :] + gates[:, None] * residual_alpha.t()))
        offsets = self.rho_b * torch.tanh(
            base_beta[None, :] + gates[:, None] * residual_beta.t())
        return gains, offsets

    def forward(self, x, phi, labels=None, class_probs=None):
        if x.shape[1] != self.bands:
            raise ValueError(f"Expected {self.bands} bands, got {x.shape[1]}")
        if labels is None and class_probs is None:
            gains = torch.exp(self.rho_a * torch.tanh(self.U @ phi[:self.K]))
            offsets = self.rho_b * torch.tanh(self.U @ phi[self.K:self.global_dim])
            gains = gains.expand(x.size(0), -1)
            offsets = offsets.expand(x.size(0), -1)
        else:
            gains, offsets = self._class_coefficients(phi)
            if labels is not None:
                gains, offsets = gains[labels.long()], offsets[labels.long()]
            else:
                if class_probs.shape != (x.size(0), self.num_classes):
                    raise ValueError("class_probs must have shape [batch, num_classes]")
                gains, offsets = class_probs @ gains, class_probs @ offsets
        return x * gains.view(-1, self.bands, 1, 1) + offsets.view(-1, self.bands, 1, 1)

    @torch.no_grad()
    def diagnostics(self, x, phi, labels=None):
        xt = self.forward(x, phi, labels=labels)
        rel = float((xt - x).norm() / x.norm().clamp_min(1e-8))
        gains, offsets = self._class_coefficients(phi)
        gain_tv = float((gains[:, 1:] - gains[:, :-1]).abs().mean(dim=1).max())
        offset_tv = float((offsets[:, 1:] - offsets[:, :-1]).abs().mean(dim=1).max())
        return {
            "relative_spectral_change": rel,
            "gain_min": float(gains.min()), "gain_max": float(gains.max()),
            "offset_min": float(offsets.min()), "offset_max": float(offsets.max()),
            "gain_tv": gain_tv, "offset_tv": offset_tv,
        }
