"""v7.1 controls: Soft LMMD and others."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

def reliability_lmmd(source_features, source_labels, target_features,
                     target_probs, stable_mask, reliability, num_classes,
                     min_samples=2, min_effective=1.5):
    """Source-vs-reliable-target class-conditional mean alignment (Soft weights)."""
    sf, tf = source_features.float(), target_features.float()
    base_weights = reliability.detach().float().clamp(0, 1) * stable_mask.detach().float()
    total = sf.sum() * 0.0
    used = 0
    for c in range(num_classes):
        sm = source_labels == c
        if int(sm.sum()) < min_samples:
            continue
        w = target_probs[:, c].detach().float() * base_weights
        if float(w.sum()) < min_effective:
            continue
        tmean = (tf * w[:, None]).sum(0) / w.sum().clamp_min(1e-8)
        smean = sf[sm].mean(0)
        total = total + (smean - tmean).square().sum()
        used += 1
    return total / used if used else total

def maturity_snapshot(memory, min_samples=12, min_classes=2):
    seen = memory.valid
    if not bool(seen.any()):
        return {'maturity': 0.0, 'eligible': 0, 'classes': 0,
                'stable_fraction': 0.0, 'flip_fraction': 1.0}
    stable = (seen & (memory.age >= max(2, memory.min_age)) &
              (memory.streak >= max(2, memory.min_streak)) &
              (memory.reliability_memory >= memory.reliability_threshold))
    eligible = int(stable.sum().item())
    labels = memory.prob_memory[stable].argmax(1) if eligible else torch.empty(0,device=seen.device,dtype=torch.long)
    classes = int(torch.unique(labels).numel())
    flip = float((memory.label_flip_count[seen] > 0).float().mean().item())
    stable_fraction = float(stable.sum().item() / max(1, seen.sum().item()))
    confidence = float(memory.reliability_memory[stable].mean().item()) if eligible else 0.0
    if eligible < min_samples or classes < min(min_classes, memory.num_classes):
        maturity = 0.0
    else:
        support_term = min(1.0, eligible / max(min_samples * 3, 1))
        class_term = min(1.0, classes / max(2, memory.num_classes // 2))
        maturity = max(0.0, min(1.0, support_term * class_term * (1.0 - 0.5 * flip) * confidence))
    return {'maturity': maturity, 'eligible': eligible, 'classes': classes,
            'stable_fraction': stable_fraction, 'flip_fraction': flip}

@torch.no_grad()
def independent_spectral_view(x, strength=0.025):
    b, channels, _, _ = x.shape
    knots = torch.randn((b, 1, min(8, channels)), device=x.device, dtype=torch.float32)
    gains = F.interpolate(knots, size=channels, mode='linear', align_corners=False)
    gains = torch.tanh(gains).view(b, channels, 1, 1)
    return (x.float() * (1.0 + strength * gains)).to(x.dtype)

def js_agreement(p, q):
    p, q = p.detach().float().clamp_min(1e-7), q.detach().float().clamp_min(1e-7)
    middle = 0.5 * (p + q)
    js = 0.5 * ((p * (p.log() - middle.log())).sum(1) + (q * (q.log() - middle.log())).sum(1))
    return (1 - js / math.log(2)).clamp(0, 1)

@torch.no_grad()
def classwise_cnt_shift(source, target, source_labels, target_probabilities,
                        source_class_means, basis, scale, max_ratio=0.12,
                        min_support=3.0, target_confidence=0.65):
    return source, 0.0, torch.zeros_like(source_class_means)

@torch.no_grad()
def source_teacher_backtrack(teacher, source, augmented, labels,
                             tolerance=0.30, rounds=3):
    from semantic_safety import judge_augmentation
    clean_logits = teacher(source).logits.detach()
    candidate = augmented
    good = torch.zeros_like(labels, dtype=torch.bool)
    for step in range(rounds + 1):
        logits = teacher(candidate).logits.detach()
        good, _ = judge_augmentation(clean_logits, logits, labels, tolerance)
        if good.all():
            break
        if step < rounds:
            candidate = torch.where(good[:,None,None,None], candidate, source + 0.5 * (candidate - source))
    return candidate, good
