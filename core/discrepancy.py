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


def _default_bandwidths(x, y):
    dxx = _sq_dist(x, x)
    dxy = _sq_dist(x, y)
    valid_dxx = dxx[torch.triu(torch.ones_like(dxx), diagonal=1) > 0]
    valid_dxy = dxy.flatten()
    valid_all = torch.cat([valid_dxx, valid_dxy]) if valid_dxx.numel() > 0 else valid_dxy
    median_dist = torch.median(valid_all).sqrt().item() if valid_all.numel() else 1.0
    return tuple(max(median_dist * scale, 1e-4) for scale in (0.5, 1.0, 2.0, 4.0))


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
        bandwidths = _default_bandwidths(x, y)
        
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
                        target_class_probs=None, target_weights=None,
                        min_effective_samples=4):
    phi = torch.zeros(transform.dim, device=source_images.device, requires_grad=True)
    if source_labels is not None:
        d_s = spectral_descriptor(transform(source_images, phi, labels=source_labels))
    else:
        d_s = spectral_descriptor(transform(source_images, phi))
    if bandwidths is None:
        bandwidths = _default_bandwidths(d_s, target_descriptor)

    if target_class_probs is None:
        valid_weights = target_weights
        if valid_weights is not None and torch.as_tensor(valid_weights).sum() <= 0:
            valid_weights = None
        loss = mk_mmd2(d_s, target_descriptor, bandwidths, y_weights=valid_weights)
    else:
        if source_labels is None:
            raise ValueError("class-conditional MMD requires labeled source samples")
        target_class_probs = torch.as_tensor(
            target_class_probs, device=target_descriptor.device, dtype=target_descriptor.dtype)
        if (target_class_probs.ndim != 2 or target_class_probs.size(0) != target_descriptor.size(0)
                or not torch.isfinite(target_class_probs).all() or (target_class_probs < 0).any()
                or (target_class_probs.sum(dim=1) <= 0).any()):
            raise ValueError("target_class_probs must be finite probability rows matching target samples")
        target_class_probs = target_class_probs / target_class_probs.sum(dim=1, keepdim=True)
        if target_weights is None:
            target_weights = torch.ones(target_descriptor.size(0), device=target_descriptor.device,
                                        dtype=target_descriptor.dtype)
        else:
            target_weights = torch.as_tensor(
                target_weights, device=target_descriptor.device, dtype=target_descriptor.dtype).reshape(-1)
            if (target_weights.numel() != target_descriptor.size(0)
                    or not torch.isfinite(target_weights).all() or (target_weights < 0).any()):
                raise ValueError("target_weights must be finite, non-negative, and match target samples")

        labels = source_labels.to(source_images.device).long()
        class_losses = []
        for c in range(target_class_probs.size(1)):
            weights = target_class_probs[:, c] * target_weights
            mass = weights.sum()
            effective_samples = mass.square() / weights.square().sum().clamp_min(1e-12)
            source_mask = labels == c
            if (mass < min_effective_samples or effective_samples < min_effective_samples
                    or not source_mask.any()):
                continue
            class_losses.append(mk_mmd2(
                d_s[source_mask], target_descriptor, bandwidths, y_weights=weights))

        if class_losses:
            loss = torch.stack(class_losses).mean()
        else:
            # No class has enough reliable support; preserve the prior marginal objective.
            valid_weights = target_weights if target_weights.sum() > 0 else None
            loss = mk_mmd2(d_s, target_descriptor, bandwidths, y_weights=valid_weights)
    grad = torch.autograd.grad(loss, phi, create_graph=False)[0]
    return grad.detach(), float(loss.detach())


@torch.no_grad()
def target_consensus_statistics(context_logits, spectral_logits, fused_logits,
                                spectral_reliability):
    logits = [torch.as_tensor(value).detach().float() for value in
              (context_logits, spectral_logits, fused_logits)]
    if (any(value.ndim != 2 or value.shape != logits[0].shape for value in logits)
            or logits[0].size(0) == 0 or logits[0].size(1) < 2
            or any(not torch.isfinite(value).all() for value in logits)):
        raise ValueError("target logits must be finite [samples, classes] matrices of equal shape")
    probabilities = [torch.softmax(value, dim=1) for value in logits]
    classes = [value.argmax(dim=1) for value in probabilities]
    agreement_count = (classes[0].eq(classes[1]).to(logits[0].dtype)
                       + classes[0].eq(classes[2]).to(logits[0].dtype)
                       + classes[1].eq(classes[2]).to(logits[0].dtype))
    agreement = agreement_count >= 2
    consensus = torch.softmax(torch.stack([F.log_softmax(value, dim=1)
                                           for value in logits]).mean(0), dim=1)
    confidence = torch.stack([value.max(dim=1).values for value in probabilities]).amin(0)
    reliability = torch.as_tensor(
        spectral_reliability, device=consensus.device, dtype=consensus.dtype).reshape(-1)
    if (reliability.numel() != consensus.size(0) or not torch.isfinite(reliability).all()):
        raise ValueError("spectral_reliability must be finite and match target samples")
    agreement_fraction = torch.where(agreement,
                                    torch.where(agreement_count >= 3,
                                                torch.ones_like(agreement_count),
                                                torch.full_like(agreement_count, 2.0 / 3.0)),
                                    torch.zeros_like(agreement_count))
    weights = (0.25 + 0.75 * agreement_fraction) * confidence * reliability.clamp(0.0, 1.0)
    weights = torch.where(agreement, weights, torch.zeros_like(weights))
    return consensus, weights, agreement


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
