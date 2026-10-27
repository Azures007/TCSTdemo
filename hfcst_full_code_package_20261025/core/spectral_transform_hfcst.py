"""Hierarchical DCT spectral transport: shared parameters + class residuals."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn


def build_dct_basis(bands: int, K: int, device=None, dtype=torch.float32) -> torch.Tensor:
    if K < 1 or K > bands:
        raise ValueError(f"K must be in [1, bands], got K={K}, bands={bands}")
    k = np.arange(K)[None, :]
    i = np.arange(bands)[:, None]
    U = np.cos(np.pi * (2 * i + 1) * k / (2 * bands)) * math.sqrt(2.0 / bands)
    U[:, 0] *= 1.0 / math.sqrt(2.0)
    return torch.as_tensor(U.astype(np.float32), device=device, dtype=dtype)


class HierarchicalSpectralTransform(nn.Module):
    def __init__(self, bands: int, num_classes: int, K: int = 8,
                 rho_a: float = 0.10, rho_b: float = 0.10):
        super().__init__()
        if K > bands:
            raise ValueError(f"K={K} exceeds bands={bands}")
        self.bands, self.num_classes, self.K = int(bands), int(num_classes), int(K)
        self.rho_a, self.rho_b = float(rho_a), float(rho_b)
        self.register_buffer("U", build_dct_basis(bands, K), persistent=True)
        self.phi_shared = nn.Parameter(torch.zeros(2 * K))
        self.delta_phi = nn.Parameter(torch.zeros(num_classes, 2 * K))
        self.register_buffer("active_classes", torch.ones(num_classes, dtype=torch.bool))

    @property
    def dim(self) -> int:
        return 2 * self.K

    @torch.no_grad()
    def set_supported_classes(self, supported):
        mask = torch.as_tensor(supported, dtype=torch.bool, device=self.active_classes.device)
        if mask.numel() != self.num_classes:
            raise ValueError("supported class mask has wrong length")
        self.active_classes.copy_(mask)
        self.delta_phi[~mask] = 0.0

    @torch.no_grad()
    def set_parameters(self, shared=None, residual=None):
        if shared is not None:
            self.phi_shared.copy_(torch.as_tensor(shared, device=self.phi_shared.device,
                                                   dtype=self.phi_shared.dtype).reshape_as(self.phi_shared))
        if residual is not None:
            self.delta_phi.copy_(torch.as_tensor(residual, device=self.delta_phi.device,
                                                  dtype=self.delta_phi.dtype).reshape_as(self.delta_phi))
            self.delta_phi[~self.active_classes] = 0.0

    def class_parameters(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long().to(self.phi_shared.device)
        if labels.min().item() < 0 or labels.max().item() >= self.num_classes:
            raise ValueError("labels outside transform class range")
        residual = self.delta_phi * self.active_classes[:, None].to(self.delta_phi.dtype)
        return self.phi_shared[None, :] + residual.index_select(0, labels)

    def _apply_phi(self, x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        # phi is [2K] or [N,2K]; output retains input shape [N,B,H,W].
        if phi.ndim == 1:
            phi = phi.unsqueeze(0).expand(x.shape[0], -1)
        alpha, beta = phi[:, :self.K], phi[:, self.K:]
        gain = torch.exp(self.rho_a * torch.tanh(alpha @ self.U.T))
        offset = self.rho_b * torch.tanh(beta @ self.U.T)
        return x * gain[:, :, None, None] + offset[:, :, None, None]

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None,
                class_probs: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.bands:
            raise ValueError(f"expected [N,{self.bands},H,W], got {tuple(x.shape)}")
        if labels is not None and class_probs is not None:
            raise ValueError("pass labels or class_probs, not both")
        if labels is not None:
            return self._apply_phi(x, self.class_parameters(labels))
        if class_probs is not None:
            probs = class_probs.to(device=x.device, dtype=x.dtype)
            if probs.shape != (x.shape[0], self.num_classes):
                raise ValueError("class_probs must have shape [N,num_classes]")
            probs = probs / probs.sum(1, keepdim=True).clamp_min(1e-8)
            outputs = []
            for c in range(self.num_classes):
                y = torch.full((x.shape[0],), c, device=x.device, dtype=torch.long)
                outputs.append(self._apply_phi(x, self.class_parameters(y)) * probs[:, c, None, None, None])
            return torch.stack(outputs, dim=0).sum(dim=0)
        return self._apply_phi(x, self.phi_shared)

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        xt = self.forward(x, labels=labels)
        if labels is None:
            phis = self.phi_shared[None, :]
        else:
            phis = self.class_parameters(labels).unique(dim=0)
        gains, offsets = [], []
        for phi in phis:
            gains.append(torch.exp(self.rho_a * torch.tanh(self.U @ phi[:self.K])))
            offsets.append(self.rho_b * torch.tanh(self.U @ phi[self.K:]))
        g = torch.stack(gains); b = torch.stack(offsets)
        return {
            "gain_min": float(g.min().item()), "gain_max": float(g.max().item()),
            "offset_min": float(b.min().item()), "offset_max": float(b.max().item()),
            "gain_tv": float(torch.abs(g[:, 1:] - g[:, :-1]).sum(dim=1).max().item()),
            "offset_tv": float(torch.abs(b[:, 1:] - b[:, :-1]).sum(dim=1).max().item()),
            "relative_spectral_change": float(((xt - x).norm() / x.norm().clamp_min(1e-8)).item()),
            "shared_norm": float(self.phi_shared.norm().item()),
            "residual_norms": [float(v.item()) for v in self.delta_phi.norm(dim=1)],
        }
