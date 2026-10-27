"""Weighted class-conditional distribution discrepancies for HFCST."""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F


def effective_sample_size(weights: torch.Tensor) -> float:
    w = torch.as_tensor(weights).float().reshape(-1).clamp_min(0)
    denom = w.square().sum()
    if w.numel() == 0 or float(denom.detach()) <= 1e-12:
        return 0.0
    return float((w.sum().square() / denom).detach().item())


def _normalise_weights(weights: torch.Tensor, n: int, device, dtype):
    if weights is None:
        return torch.full((n,), 1.0 / max(n, 1), device=device, dtype=dtype)
    w = torch.as_tensor(weights, device=device, dtype=dtype).reshape(-1).clamp_min(0)
    if w.numel() != n:
        raise ValueError(f"weights length {w.numel()} != samples {n}")
    s = w.sum()
    if float(s.detach()) <= 1e-12:
        return torch.zeros_like(w)
    return w / s


def _rbf(x, y, bandwidth):
    d2 = torch.cdist(x, y, p=2).square().clamp_min(0)
    return torch.exp(-d2 / (2.0 * bandwidth.clamp_min(1e-8)))


def weighted_mmd(x: torch.Tensor, y: torch.Tensor, target_weights: torch.Tensor | None = None,
                 source_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Biased multi-scale RBF MMD^2 with optional normalized sample weights."""
    x = x.reshape(x.shape[0], -1).float()
    y = y.reshape(y.shape[0], -1).float()
    if x.shape[1] != y.shape[1]:
        raise ValueError("MMD inputs must have the same feature dimension")
    if x.shape[0] == 0 or y.shape[0] == 0:
        return (x.sum() + y.sum()) * 0.0
    wx = _normalise_weights(source_weights, x.shape[0], x.device, x.dtype)
    wy = _normalise_weights(target_weights, y.shape[0], y.device, y.dtype)
    if float(wx.sum().detach()) == 0 or float(wy.sum().detach()) == 0:
        return (x.sum() + y.sum()) * 0.0
    z = torch.cat([x.detach(), y.detach()], dim=0)
    if z.shape[0] > 512:
        z = z[torch.linspace(0, z.shape[0]-1, 512, device=z.device).long()]
    dist = torch.pdist(z, p=2).square()
    positive = dist[dist > 1e-10]
    base = positive.median() if positive.numel() else torch.tensor(1.0, device=x.device)
    base = base.detach().clamp_min(1e-5)
    kxx = sum(_rbf(x, x, base * scale) for scale in (0.5, 1.0, 2.0)) / 3.0
    kyy = sum(_rbf(y, y, base * scale) for scale in (0.5, 1.0, 2.0)) / 3.0
    kxy = sum(_rbf(x, y, base * scale) for scale in (0.5, 1.0, 2.0)) / 3.0
    value = wx @ kxx @ wx + wy @ kyy @ wy - 2.0 * wx @ kxy @ wy
    return value.clamp_min(0.0)


def spectral_descriptor(patches: torch.Tensor) -> torch.Tensor:
    """Center spectrum shape, first/second differences and log amplitude."""
    x = patches.float()
    spec = x[:, :, x.shape[2] // 2, x.shape[3] // 2]
    mean = spec.mean(dim=1, keepdim=True)
    std = spec.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    shape = (spec - mean) / std
    d1 = shape[:, 1:] - shape[:, :-1]
    d2 = d1[:, 1:] - d1[:, :-1]
    log_amp = torch.log(torch.linalg.vector_norm(spec, dim=1, keepdim=True).clamp_min(1e-8))
    return torch.cat([shape, d1, d2, log_amp], dim=1)


def class_transport_from_embeddings(source_desc: torch.Tensor, source_feat: torch.Tensor,
                                    source_labels: torch.Tensor, target_desc: torch.Tensor,
                                    target_feat: torch.Tensor, q: torch.Tensor, reliability: torch.Tensor,
                                    num_classes: int, lambda_spec: float = 1.0,
                                    lambda_feat: float = 1.0, min_effective: float = 8.0,
                                    min_source: int = 4, residual_penalty: torch.Tensor | None = None):
    """Compute equal-weight mean of reliable class-conditional MMD terms."""
    source_labels = source_labels.long()
    q = q.float()
    reliability = reliability.float().reshape(-1).clamp_min(0)
    per_class, effective, components = {}, {}, {}
    for c in range(num_classes):
        si = torch.nonzero(source_labels == c, as_tuple=False).flatten()
        weights = reliability * q[:, c]
        neff = effective_sample_size(weights)
        effective[c] = neff
        if si.numel() < min_source or neff < min_effective or float(weights.sum().detach()) <= 1e-8:
            continue
        ds = weighted_mmd(source_desc.index_select(0, si), target_desc, target_weights=weights)
        df = weighted_mmd(source_feat.index_select(0, si), target_feat, target_weights=weights)
        per_class[c] = lambda_spec * ds + lambda_feat * df
        components[c] = {"d_spec": float(ds.detach().item()), "d_feat": float(df.detach().item()),
                         "effective_target_n": neff, "source_n": int(si.numel())}
    if per_class:
        total = torch.stack(list(per_class.values())).mean()
    else:
        total = (source_desc.sum() + source_feat.sum()) * 0.0
    if residual_penalty is not None:
        total = total + residual_penalty
    return total, {"per_class": components, "effective_target_n": effective,
                   "valid_classes": sorted(per_class), "n_valid_classes": len(per_class)}
