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


def mk_mmd2(x, y, bandwidths=None, y_weights=None):
    dxx = _sq_dist(x, x); dyy = _sq_dist(y, y); dxy = _sq_dist(x, y)
    weights = None
    if y_weights is not None:
        weights = torch.as_tensor(y_weights, device=y.device, dtype=y.dtype).reshape(-1)
        if (weights.numel() != y.size(0) or not torch.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError("y_weights must be finite, non-negative, and have positive mass")
        weights = weights / weights.sum()
    
    # 如果没有指定带宽，使用中位数启发式（自动适应高维空间）
    if bandwidths is None:
        # 取源域和目标域真实样本对之间的非零中位数距离
        # A11 fix: include cross-domain distances in the median estimate.
        # Previously only source internal distances were used, which can underestimate
        # the true scale when domain shift is large (over-saturating the kernel).
        valid_dxx = dxx[torch.triu(torch.ones_like(dxx), diagonal=1) > 0]
        valid_dxy = dxy.flatten()
        valid_all = torch.cat([valid_dxx, valid_dxy]) if valid_dxx.numel() > 0 else valid_dxy
        if valid_all.numel() > 0:
            median_dist = torch.median(valid_all).sqrt().item()
        else:
            median_dist = 1.0
        bandwidths = (median_dist * 0.5, median_dist, median_dist * 2.0, median_dist * 4.0)
        bandwidths = tuple(max(b, 1e-4) for b in bandwidths)
        
    total = x.new_tensor(0.0)
    for bw in bandwidths:
        kxx = torch.exp(-dxx / (2*bw*bw))
        kyy = torch.exp(-dyy / (2*bw*bw))
        kxy = torch.exp(-dxy / (2*bw*bw))
        if weights is None:
            total = total + (kxx.mean() + kyy.mean() - 2 * kxy.mean())
        else:
            total = total + (kxx.mean()
                             + (kyy * weights[:, None] * weights[None, :]).sum()
                             - 2 * (kxy * weights[None, :]).sum() / x.size(0))
    return total / len(bandwidths)


def domain_gradient_mmd(transform, source_images, target_descriptor,
                        bandwidths=None, source_labels=None,
                        target_class_probs=None):
    phi = torch.zeros(transform.dim, device=source_images.device, requires_grad=True)
    if source_labels is not None:
        d_s = spectral_descriptor(transform(source_images, phi, labels=source_labels))
    else:
        d_s = spectral_descriptor(transform(source_images, phi))
    loss = mk_mmd2(d_s, target_descriptor, bandwidths)
    if target_class_probs is None:
        grad = torch.autograd.grad(loss, phi, create_graph=False)[0]
        return grad.detach(), float(loss.detach())
    if source_labels is None or not hasattr(transform, "class_reliability"):
        raise ValueError("class-conditional MMD requires labeled source samples and a hierarchical transform")
    target_class_probs = torch.as_tensor(
        target_class_probs, device=target_descriptor.device, dtype=target_descriptor.dtype)
    if target_class_probs.shape != (target_descriptor.size(0), transform.num_classes):
        raise ValueError("target_class_probs must have shape [target samples, num_classes]")
    if (not torch.isfinite(target_class_probs).all() or (target_class_probs < 0).any()
            or (target_class_probs.sum(dim=1) <= 0).any()):
        raise ValueError("target_class_probs must be finite, non-negative probability rows")

    grad = torch.zeros_like(phi)
    global_grad = torch.autograd.grad(loss, phi, retain_graph=True)[0]
    grad[:transform.global_dim] = global_grad[:transform.global_dim]
    labels = source_labels.to(source_images.device).long()
    classes, residual_k = transform.num_classes, transform.residual_k
    reliability = transform.class_reliability.to(source_images.device)
    normalizer = reliability.sum().clamp_min(1.0)
    for c in range(classes):
        weights = target_class_probs[:, c]
        source_mask = labels == c
        if reliability[c] <= 0 or not source_mask.any() or weights.sum() <= 0:
            continue
        class_loss = mk_mmd2(d_s[source_mask], target_descriptor,
                             bandwidths, y_weights=weights)
        class_grad = torch.autograd.grad(class_loss, phi, retain_graph=True)[0]
        scale = reliability[c] / normalizer
        alpha = transform.global_dim + c * residual_k
        beta = transform.global_dim + classes * residual_k + c * residual_k
        grad[alpha:alpha + residual_k] = scale * class_grad[alpha:alpha + residual_k]
        grad[beta:beta + residual_k] = scale * class_grad[beta:beta + residual_k]
    return grad.detach(), float(loss.detach())


@torch.no_grad()
def sliced_wasserstein(x, y, num_projections=64, seed=0):
    """SWD with unequal sample counts (empirical quantile interpolation)."""
    device = x.device
    if x.size(0) == 0 or y.size(0) == 0:
        return 0.0
    g = torch.Generator(device=device).manual_seed(int(seed))
    D = x.shape[1]
    proj = F.normalize(torch.randn(D, num_projections, generator=g, device=device), dim=0)
    px, _ = torch.sort(x @ proj, dim=0)   # [Nx, L]
    py, _ = torch.sort(y @ proj, dim=0)   # [Ny, L]
    nx, ny = px.size(0), py.size(0)
    if nx == ny:
        return float((px - py).abs().mean())
    # 在 [0,1] 网格上取 empirical quantile，做插值
    qs = torch.linspace(0.0, 1.0, steps=256, device=device, dtype=px.dtype)
    def _quantile(sorted_vals, q):
        # sorted_vals: [N, L], q: [Q]
        n = sorted_vals.size(0)
        idx = q * (n - 1)
        lo = torch.floor(idx).long().clamp(0, n - 1)
        hi = torch.ceil(idx).long().clamp(0, n - 1)
        w = (idx - lo.float()).view(-1, 1)
        return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w   # [Q, L]
    qx = _quantile(px, qs)   # [Q, L]
    qy = _quantile(py, qs)   # [Q, L]
    return float((qx - qy).abs().mean())
