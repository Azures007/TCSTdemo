"""Risk-budgeted candidate validation for HFCST transport transactions."""
from __future__ import annotations
import numpy as np
import torch


@torch.no_grad()
def source_logits(model, transform, patches, labels, device, batch_size=128):
    model.eval()
    outs = []
    for i in range(0, len(patches), batch_size):
        xb = torch.as_tensor(patches[i:i+batch_size], dtype=torch.float32, device=device)
        yb = torch.as_tensor(labels[i:i+batch_size], dtype=torch.long, device=device)
        outs.append(model(transform(xb, labels=yb)).logits.float().cpu())
    return torch.cat(outs, dim=0) if outs else torch.empty((0, getattr(model, "num_classes", 0)))


def source_risk_from_logits(before_logits, after_logits, labels, num_classes,
                            margin_tolerance=0.05):
    y = torch.as_tensor(labels, dtype=torch.long).cpu()
    if len(y) == 0:
        return {"available": False, "reason": "empty_source_calibration", "new_error_rate": None,
                "worst_class_new_error": None, "margin_loss": None}
    b = before_logits.detach().float().cpu()
    a = after_logits.detach().float().cpu()
    if len(b) != len(y) or len(a) != len(y):
        raise ValueError("source logits/labels size mismatch")
    p0, p1 = b.argmax(1), a.argmax(1)
    correct0, wrong1 = p0.eq(y), p1.ne(y)
    new_error = correct0 & wrong1
    per_class = {}
    for c in range(num_classes):
        mask = y.eq(c)
        per_class[str(c)] = float((new_error & mask).sum().item() / max(1, int(mask.sum().item()))) if mask.any() else None
    def margin(z):
        true = z.gather(1, y[:, None]).squeeze(1)
        other = z.clone(); other.scatter_(1, y[:, None], -1e9)
        return true - other.max(1).values
    m0, m1 = margin(b), margin(a)
    margin_loss = torch.relu(m0 - m1 - float(margin_tolerance)).mean().item()
    return {"available": True, "new_error_rate": float(new_error.float().mean().item()),
            "worst_class_new_error": float(max((v for v in per_class.values() if v is not None), default=0.0)),
            "per_class_new_error": per_class, "margin_loss": float(margin_loss),
            "accuracy_before": float(p0.eq(y).float().mean().item()),
            "accuracy_after": float(p1.eq(y).float().mean().item())}


def spectral_plausibility(diag, cfg):
    limits = {
        "gain_max_deviation": float(cfg.get("gain_max_deviation", 0.15)),
        "offset_max_abs": float(cfg.get("offset_max_abs", 0.15)),
        "gain_tv_max": float(cfg.get("gain_tv_max", 0.36)),
        "offset_tv_max": float(cfg.get("offset_tv_max", 0.18)),
        "relative_change_max": float(cfg.get("relative_change_max", 0.15)),
    }
    reasons = []
    for key in ("gain_min", "gain_max", "offset_min", "offset_max", "gain_tv", "offset_tv", "relative_spectral_change"):
        if key not in diag or not np.isfinite(diag[key]):
            reasons.append(f"nonfinite_or_missing_{key}")
    if reasons:
        return False, reasons
    if diag["gain_min"] <= 0:
        reasons.append("gain_not_positive")
    if max(abs(diag["gain_min"] - 1.0), abs(diag["gain_max"] - 1.0)) > limits["gain_max_deviation"]:
        reasons.append("gain_deviation_exceeded")
    if max(abs(diag["offset_min"]), abs(diag["offset_max"])) > limits["offset_max_abs"]:
        reasons.append("offset_exceeded")
    if diag["gain_tv"] > limits["gain_tv_max"]:
        reasons.append("gain_tv_exceeded")
    if diag["offset_tv"] > limits["offset_tv_max"]:
        reasons.append("offset_tv_exceeded")
    if diag["relative_spectral_change"] > limits["relative_change_max"]:
        reasons.append("relative_change_exceeded")
    return not reasons, reasons


def candidate_acceptance(before_score, after_score, bootstrap_improvements, risk,
                         spectral_ok, target_flip_rate, cfg, target_block_stability=None):
    reasons = []
    if before_score is None or after_score is None or not np.isfinite(before_score) or not np.isfinite(after_score):
        reasons.append("transport_score_unavailable")
    else:
        absolute_min = float(cfg.get("minimum_transport_improvement", 1e-5))
        relative_min = float(cfg.get("minimum_relative_improvement", 1e-4))
        improvement = before_score - after_score
        if improvement <= max(absolute_min, abs(before_score) * relative_min):
            reasons.append("transport_not_improved")
    boot = np.asarray(bootstrap_improvements, dtype=float)
    boot = boot[np.isfinite(boot)]
    if len(boot) < int(cfg.get("minimum_bootstrap_replicates", 5)):
        reasons.append("bootstrap_unavailable")
        ci_low = None
    else:
        ci_low = float(np.quantile(boot, 1.0 - float(cfg.get("bootstrap_confidence", 0.90))))
        if ci_low <= 0:
            reasons.append("bootstrap_improvement_not_positive")
    if not risk.get("available", False):
        reasons.append("source_calibration_unavailable")
    else:
        if risk["new_error_rate"] > float(cfg.get("max_new_error_rate", 0.02)):
            reasons.append("source_new_error_budget_exceeded")
        if risk["worst_class_new_error"] > float(cfg.get("max_class_new_error_rate", 0.10)):
            reasons.append("source_class_error_budget_exceeded")
        if risk["margin_loss"] > float(cfg.get("max_margin_loss", 0.20)):
            reasons.append("source_margin_loss_budget_exceeded")
    if not spectral_ok:
        reasons.append("spectral_plausibility_failed")
    if target_flip_rate is None or not np.isfinite(target_flip_rate):
        reasons.append("target_pseudo_stability_unavailable")
    elif target_flip_rate > float(cfg.get("max_target_pseudo_flip_rate", 0.25)):
        reasons.append("target_pseudo_stability_degraded")
    if target_block_stability is not None:
        if not target_block_stability.get("available", False):
            reasons.append("target_block_stability_unavailable")
        elif target_block_stability.get("min_block_view_agreement") is not None and \
                target_block_stability["min_block_view_agreement"] < float(cfg.get("min_block_view_agreement", 0.40)):
            reasons.append("target_block_view_agreement_too_low")
    return {"accepted": not reasons, "reasons": reasons, "bootstrap_ci_low": ci_low,
            "bootstrap_n": int(len(boot)),
            "transport_before": before_score, "transport_after": after_score,
            "transport_improvement": (before_score - after_score) if before_score is not None and after_score is not None else None,
            "target_pseudo_flip_rate": target_flip_rate, "source_risk": risk}
