"""Risk-budgeted backtracking line search (v8.0.1).

v8.0.1 修复:
- 返回 accepted_candidate（含 Vc），供 RS-CRW 使用
- 加入光谱合理性检查 (Gain>0, |a-1|_inf, |b|_inf, NaN)
- SWD 使用相对改善阈值 tau_swd_rel
"""
from __future__ import annotations
import numpy as np
import torch

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


def _check_spectral_plausibility(diag, gain_max_dev, offset_max_abs,
                                 gain_tv_max=0.36, offset_tv_max=0.18,
                                 rel_change_max=0.15):
    """Return (ok, reason) for spectral plausibility."""
    g_min = diag.get("gain_min", 1.0)
    g_max = diag.get("gain_max", 1.0)
    o_min = diag.get("offset_min", 0.0)
    o_max = diag.get("offset_max", 0.0)
    if not np.isfinite(g_min) or not np.isfinite(g_max):
        return False, "NaN/Inf in gain"
    if not np.isfinite(o_min) or not np.isfinite(o_max):
        return False, "NaN/Inf in offset"
    if g_min <= 0:
        return False, "gain not strictly positive"
    if max(abs(g_min - 1.0), abs(g_max - 1.0)) > gain_max_dev:
        return False, f"|a-1|_inf > {gain_max_dev}"
    if max(abs(o_min), abs(o_max)) > offset_max_abs:
        return False, f"|b|_inf > {offset_max_abs}"
    gain_tv = diag.get("gain_tv", 0.0)
    offset_tv = diag.get("offset_tv", 0.0)
    rel_change = diag.get("relative_spectral_change", 0.0)
    # A6 fix: TV thresholds were 2.0/1.0, unreachable because rho_a<=0.15 limits
    # gain_tv to ~0.32 and offset_tv to ~0.30. Tighten to 60% of the mathematical bound.
    # With rho_a=0.15: max gain swing per adjacent pair <= 2*0.30=0.60, so TV cap 0.36.
    # With rho_a=0.10: max gain swing <= 2*0.21=0.42, so TV cap 0.25.
    if gain_tv > gain_tv_max:
        return False, f"Gain TV too high ({gain_tv:.3f} > {gain_tv_max})"
    if offset_tv > offset_tv_max:
        return False, f"Offset TV too high ({offset_tv:.3f} > {offset_tv_max})"
    if rel_change > rel_change_max:
        return False, f"Relative spectral change too high ({rel_change:.3f} > {rel_change_max})"
    return True, "ok"


def backtracking_search(model, transform, d_star,
                        source_cal_patches, source_cal_labels,
                        target_cal_patches, eta_candidates,
                        tau_sem, tau_flip, tau_class_flip,
                        tau_swd_rel, device, swd_seed=0,
                        gain_max_dev=0.5, offset_max_abs=0.5,
                        gain_tv_max=0.36, offset_tv_max=0.18,
                        rel_change_max=0.15, rsem_delta=0.05,
                        domain_cal_patches=None):
    """Return (eta_star, phi_star, diagnostics).

    diagnostics["accepted_candidate"] = the candidate dict that was ACCEPTED,
    or None if all rejected. RS-CRW must use this to read Vc.
    """
    tau_sem = float(tau_sem); tau_flip = float(tau_flip)
    tau_class_flip = float(tau_class_flip)
    tau_swd_rel = float(tau_swd_rel)
    eta_candidates = [float(e) for e in eta_candidates]

    model.eval()
    d_star_t = torch.from_numpy(d_star).to(device)
    dim = transform.dim

    with torch.no_grad():
        pred0, correct0, margin0 = _eval_margins(
            model, source_cal_patches, source_cal_labels, device)
        # B4 fix: 用全部校准样本，不只前 64（HyRANK 144 个只用 44% 会低估真实 SWD）
        # T9: SWD 用独立的 domain calibration（如果提供），而非 BA-CS 边界样本
        _src_swd = domain_cal_patches if domain_cal_patches is not None else source_cal_patches
        src_desc0 = spectral_descriptor(torch.from_numpy(_src_swd).to(device))
        tgt_desc = spectral_descriptor(torch.from_numpy(target_cal_patches).to(device))
        swd0 = sliced_wasserstein(src_desc0, tgt_desc, seed=swd_seed)
        swd0_safe = max(float(swd0), 1e-8)

    n_classes = int(source_cal_labels.max()) + 1
    cal_counts = np.bincount(source_cal_labels, minlength=n_classes).astype(np.float64)

    diagnostics = {"swd_before": float(swd0), "candidates": [],
                   "cal_class_counts": cal_counts.tolist(),
                   "accepted_candidate": None}
    best_rejected = None

    for eta in eta_candidates:
        phi = eta * d_star_t
        with torch.no_grad():
            logits_batches = []
            for start in range(0, len(source_cal_patches), 128):
                xb = torch.from_numpy(source_cal_patches[start:start+128]).to(device)
                yb = torch.from_numpy(source_cal_labels[start:start+128]).long().to(device)
                logits_batches.append(model(transform(xb, phi, labels=yb)).logits.float())
            logits1 = torch.cat(logits_batches, 0).cpu()
            correct1 = logits1.argmax(1).eq(torch.from_numpy(source_cal_labels).long())

            new_err = (correct0 & (~correct1))
            V = float(new_err.float().mean().item())
            y_t = torch.from_numpy(source_cal_labels).long()
            Vc = np.zeros(n_classes, dtype=np.float64)
            for c in range(n_classes):
                if cal_counts[c] > 0:
                    Vc[c] = float((new_err & (y_t == c)).float().sum().item()) / cal_counts[c]
            Vmax = float(Vc.max()) if Vc.size else 0.0

            # P2: 逐类 margin 变化（判断 MMD 降低是否伴随某些类恶化）
            true_logit1 = logits1.gather(1, y_t[:, None]).squeeze(1)
            other1 = logits1.clone(); other1.scatter_(1, y_t[:, None], -1e9)
            margin1 = true_logit1 - other1.max(1).values
            _margin_delta = (margin1 - margin0).numpy()  # 正=改善，负=恶化
            _per_class_margin = {}
            for c in range(n_classes):
                m = (y_t == c).numpy()
                if m.sum() > 0:
                    _per_class_margin[int(c)] = {
                        "mean_delta": float(_margin_delta[m].mean()),
                        "min_delta": float(_margin_delta[m].min()),
                        "frac_neg": float((_margin_delta[m] < 0).mean()),
                        "n": int(m.sum()),
                    }

            true_logit1 = logits1.gather(1, y_t[:, None]).squeeze(1)
            other1 = logits1.clone(); other1.scatter_(1, y_t[:, None], -1e9)
            margin1 = true_logit1 - other1.max(1).values
            Rsem = float((torch.relu((margin0 - margin1) - rsem_delta) ** 2).mean().item())

            _src_swd1 = domain_cal_patches if domain_cal_patches is not None else source_cal_patches
            src_desc1 = spectral_descriptor(
                transform(torch.from_numpy(_src_swd1).to(device), phi,
                          labels=None))
            swd1 = sliced_wasserstein(src_desc1, tgt_desc, seed=swd_seed)
            swd_drop = float(swd0 - swd1)
            swd_drop_rel = float(swd_drop / swd0_safe)
            spectral_diag = transform.diagnostics(
                torch.from_numpy(source_cal_patches).to(device), phi,
                labels=torch.from_numpy(source_cal_labels).long().to(device))

            spec_ok, spec_reason = _check_spectral_plausibility(
                spectral_diag, gain_max_dev, offset_max_abs,
                gain_tv_max=gain_tv_max, offset_tv_max=offset_tv_max,
                rel_change_max=rel_change_max)

            diag = {"eta": float(eta), "V": V, "Vmax": Vmax, "Rsem": Rsem,
                    "swd_before": float(swd0), "swd_after": float(swd1),
                    "swd_drop": swd_drop, "swd_drop_rel": swd_drop_rel,
                    "Vc": Vc.tolist(),
                    "per_class_margin": _per_class_margin,
                    "phi_norm": float(phi.norm()), "spectral": spectral_diag,
                    "spectral_ok": spec_ok, "spectral_reason": spec_reason}
            diagnostics["candidates"].append(diag)

            reasons = []
            if V > tau_flip: reasons.append(f"V>{tau_flip}")
            if Vmax > tau_class_flip: reasons.append(f"Vmax>{tau_class_flip}")
            if Rsem > tau_sem: reasons.append(f"Rsem>{tau_sem}")
            if swd_drop_rel < tau_swd_rel: reasons.append(f"swd_rel<{tau_swd_rel}")
            if not spec_ok: reasons.append(f"spectral:{spec_reason}")

            if not reasons:
                print(f"  [risk] eta={eta:.4f} V={V:.4f} Vmax={Vmax:.3f} "
                      f"Rsem={Rsem:.4f} swd_rel={swd_drop_rel:+.2e} "
                      f"spec={spec_reason} -> ACCEPT", flush=True)
                diagnostics["accepted_candidate"] = diag
                return float(eta), phi, diagnostics

            print(f"  [risk] eta={eta:.4f} V={V:.4f} Vmax={Vmax:.3f} "
                  f"Rsem={Rsem:.4f} swd_rel={swd_drop_rel:+.2e} "
                  f"-> REJECT({','.join(reasons)})", flush=True)
            if best_rejected is None:
                best_rejected = diag

    diagnostics["best_rejected"] = best_rejected
    return 0.0, torch.zeros(dim, device=device), diagnostics
