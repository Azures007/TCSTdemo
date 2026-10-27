"""End-to-end HFCST-v9 training with spatial isolation and transactional adaptation."""
from __future__ import annotations
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from core.data_hfcst import (load_cube, load_gt, align_cubes, robust_source_scale,
                             global_spatial_split, global_target_split, build_patches,
                             audit_no_cross_split_patch_overlap)
from core.model_hfcst import HFCSTModel
from core.spectral_transform_hfcst import HierarchicalSpectralTransform
from core.discrepancy_hfcst import (spectral_descriptor, class_transport_from_embeddings,
                                    effective_sample_size)
from core.task_geometry_hfcst import select_constraint_samples, compute_margin_gradients
from core.transport_solver_hfcst import solve_margin_qp
from core.risk_controller_hfcst import (source_logits, source_risk_from_logits,
                                        spectral_plausibility, candidate_acceptance)


def _seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _resolve_path(path, root):
    raw = os.path.expandvars(str(path))
    if "$" in raw:
        raise ValueError(f"Unexpanded environment variable in path: {path}. Set DATA_ROOT or edit config.")
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (root / p).resolve()


def _jsonable(obj):
    if isinstance(obj, dict): return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, torch.Tensor): return obj.detach().cpu().tolist()
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj): return None
    return obj


def _append_jsonl(path, event):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(event), ensure_ascii=False, allow_nan=False) + "\n")


def _labels_at(gt, coords):
    if len(coords) == 0: return np.empty(0, dtype=np.int64)
    # Dataset convention in TCST: background=0 and foreground classes=1..C.
    return gt[coords[:, 0], coords[:, 1]].astype(np.int64) - 1


def _tensor(patches, device):
    return torch.as_tensor(patches, dtype=torch.float32, device=device)


def _eval_logits(model, patches, device, batch_size=128):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            out.append(model(_tensor(patches[i:i+batch_size], device)).logits.float().cpu())
    return torch.cat(out, dim=0) if out else torch.empty((0, model.num_classes))


def _source_val_metrics(model, patches, labels, device, num_classes):
    logits = _eval_logits(model, patches, device)
    if len(labels) == 0:
        return {"available": False, "n": 0, "nll": None, "oa": None, "aa": None}
    y = torch.as_tensor(labels, dtype=torch.long)
    nll = float(F.cross_entropy(logits, y).item())
    pred = logits.argmax(1)
    oa = float(pred.eq(y).float().mean().item())
    class_counts = {str(c): int((y == c).sum().item()) for c in range(num_classes)}
    per_class = {str(c): float(pred[y == c].eq(c).float().mean().item())
                 for c in range(num_classes) if (y == c).any()}
    missing_classes = [c for c in range(num_classes) if class_counts[str(c)] == 0]
    return {"available": True, "n": int(len(y)), "nll": nll, "oa": oa,
            "aa": float(np.mean(list(per_class.values()))) if per_class else None,
            "per_class_accuracy": per_class, "class_counts": class_counts,
            "missing_classes": missing_classes, "class_coverage_complete": not missing_classes}


def _supervised_loss(output, y, cfg):
    return (F.cross_entropy(output.logits, y, label_smoothing=float(cfg.get("label_smoothing", 0.0)))
            + float(cfg.get("lambda_context", 0.5)) * F.cross_entropy(output.context_logits, y)
            + float(cfg.get("lambda_spectral", 0.5)) * F.cross_entropy(output.spectral_logits, y))


def _ema_update(teacher, student, decay):
    with torch.no_grad():
        t_params = dict(teacher.named_parameters())
        for name, p in student.named_parameters():
            t_params[name].mul_(decay).add_(p.detach(), alpha=1.0-decay)
        t_buffers = dict(teacher.named_buffers())
        for name, b in student.named_buffers():
            if name in t_buffers:
                if b.is_floating_point(): t_buffers[name].mul_(decay).add_(b.detach(), alpha=1.0-decay)
                else: t_buffers[name].copy_(b)


def _teacher_assignments(teacher, patches, device, num_classes, temperature=1.0,
                         perturb_std=0.01, seed=0, batch_size=128):
    if len(patches) == 0:
        return {"q": np.empty((0, num_classes), dtype=np.float32), "r": np.empty(0, dtype=np.float32),
                "confidence": np.empty(0, dtype=np.float32), "view_consistency": np.empty(0, dtype=np.float32),
                "view_pred0": np.empty(0, dtype=np.int64), "view_pred1": np.empty(0, dtype=np.int64),
                "pred_dist": [0.0] * num_classes}
    teacher.eval(); p0s, p1s = [], []
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            xb = _tensor(patches[i:i+batch_size], device)
            # The augmentation is intentionally light and does not use target labels.
            noise = torch.randn(xb.shape, generator=gen, dtype=xb.dtype, device="cpu").to(device)
            xa = xb * (1.0 + float(perturb_std) * noise)
            p0 = torch.softmax(teacher(xb).logits.float() / max(float(temperature), 1e-4), dim=1)
            p1 = torch.softmax(teacher(xa).logits.float() / max(float(temperature), 1e-4), dim=1)
            p0s.append(p0.cpu()); p1s.append(p1.cpu())
    p0, p1 = torch.cat(p0s), torch.cat(p1s)
    q = 0.5 * (p0 + p1)
    entropy = -(q * torch.log(q.clamp_min(1e-8))).sum(dim=1)
    confidence = (1.0 - entropy / math.log(max(2, num_classes))).clamp(0, 1)
    view_consistency = (1.0 - 0.5 * torch.abs(p0 - p1).sum(dim=1)).clamp(0, 1)
    reliability = confidence * view_consistency
    pred_dist = torch.bincount(q.argmax(1), minlength=num_classes).float()
    pred_dist = (pred_dist / pred_dist.sum().clamp_min(1)).tolist()
    return {"q": q.numpy().astype(np.float32), "r": reliability.numpy().astype(np.float32),
            "confidence": confidence.numpy().astype(np.float32),
            "view_consistency": view_consistency.numpy().astype(np.float32),
            "view_pred0": p0.argmax(1).numpy().astype(np.int64),
            "view_pred1": p1.argmax(1).numpy().astype(np.int64),
            "pred_dist": pred_dist}


def _block_assignment_stability(assignment, block_ids):
    blocks = np.asarray(block_ids, dtype=np.int64).reshape(-1)
    p0 = np.asarray(assignment.get("view_pred0", []), dtype=np.int64).reshape(-1)
    p1 = np.asarray(assignment.get("view_pred1", []), dtype=np.int64).reshape(-1)
    if len(blocks) == 0 or len(p0) != len(blocks) or len(p1) != len(blocks):
        return {"available": False, "mean_view_agreement": None, "min_block_view_agreement": None,
                "per_block": {}}
    per_block = {}
    for b in np.unique(blocks):
        ix = np.flatnonzero(blocks == b)
        per_block[str(int(b))] = {"n": int(len(ix)), "view_agreement": float(np.mean(p0[ix] == p1[ix]))}
    vals = [v["view_agreement"] for v in per_block.values()]
    return {"available": True, "mean_view_agreement": float(np.mean(vals)),
            "min_block_view_agreement": float(np.min(vals)), "per_block": per_block}


def _block_flip_rates(before_predictions, after_predictions, block_ids):
    before = np.asarray(before_predictions).reshape(-1)
    after = np.asarray(after_predictions).reshape(-1)
    blocks = np.asarray(block_ids).reshape(-1)
    if len(before) == 0 or len(before) != len(after) or len(before) != len(blocks):
        return {"available": False, "max_block_flip_rate": None, "per_block": {}}
    per_block = {}
    for b in np.unique(blocks):
        ix = np.flatnonzero(blocks == b)
        per_block[str(int(b))] = float(np.mean(before[ix] != after[ix]))
    return {"available": True, "max_block_flip_rate": float(max(per_block.values(), default=0.0)),
            "per_block": per_block}


def _target_cache(model, patches, q, r, block_ids, device):
    model.eval(); feats, logits = [], []
    with torch.no_grad():
        for i in range(0, len(patches), 128):
            out = model(_tensor(patches[i:i+128], device))
            feats.append(out.features.float().cpu()); logits.append(out.logits.float().cpu())
    features = torch.cat(feats, dim=0) if feats else torch.empty((0, model.feature_dim))
    all_patches = _tensor(patches, device)
    with torch.no_grad(): desc = spectral_descriptor(all_patches).cpu()
    return {"features": features, "descriptors": desc,
            "q": torch.as_tensor(q, dtype=torch.float32), "r": torch.as_tensor(r, dtype=torch.float32),
            "block_ids": np.asarray(block_ids, dtype=np.int64),
            "logits": torch.cat(logits, dim=0) if logits else torch.empty((0, model.num_classes))}


def _residual_penalty(transform, cfg):
    lam = float(cfg.get("lambda_residual", 0.01))
    active = transform.active_classes[:, None].to(transform.delta_phi.dtype)
    return lam * (transform.delta_phi * active).square().sum()


def _transport_tensor(transform, model, source_x, source_y, target_cache, cfg):
    tx = transform(source_x, labels=source_y)
    out = model(tx)
    src_desc = spectral_descriptor(tx)
    loss, diag = class_transport_from_embeddings(
        src_desc, out.features, source_y, target_cache["descriptors"].to(source_x.device),
        target_cache["features"].to(source_x.device), target_cache["q"].to(source_x.device),
        target_cache["r"].to(source_x.device), model.num_classes,
        lambda_spec=float(cfg.get("lambda_spec", 1.0)),
        lambda_feat=float(cfg.get("lambda_feat", 1.0)),
        min_effective=float(cfg.get("min_effective_target_n", 8.0)),
        min_source=int(cfg.get("min_source_per_class", 4)),
        residual_penalty=_residual_penalty(transform, cfg))
    return loss, diag


def _source_embeddings(transform, model, source_x, source_y):
    model.eval()
    tx = transform(source_x, labels=source_y)
    out = model(tx)
    return spectral_descriptor(tx).detach(), out.features.detach()


def _score_embeddings(source_desc, source_feat, source_y, target_cache, transform, cfg, indices=None,
                      residual_penalty_value=None):
    if indices is None:
        indices = np.arange(len(target_cache["q"]))
    ids = torch.as_tensor(indices, dtype=torch.long)
    dev = source_desc.device
    loss, diag = class_transport_from_embeddings(
        source_desc, source_feat, source_y,
        target_cache["descriptors"].to(dev).index_select(0, ids.to(dev)),
        target_cache["features"].to(dev).index_select(0, ids.to(dev)),
        target_cache["q"].to(dev).index_select(0, ids.to(dev)),
        target_cache["r"].to(dev).index_select(0, ids.to(dev)),
        int(cfg["num_classes"]), lambda_spec=float(cfg.get("lambda_spec", 1.0)),
        lambda_feat=float(cfg.get("lambda_feat", 1.0)),
        min_effective=float(cfg.get("min_effective_target_n", 8.0)),
        min_source=int(cfg.get("min_source_per_class", 4)),
        residual_penalty=(_residual_penalty(transform, cfg).detach() if residual_penalty_value is None
                          else torch.as_tensor(float(residual_penalty_value), device=dev)))
    return float(loss.detach().cpu().item()), diag


def _bootstrap_improvements(before_emb, after_emb, source_y, target_cache, transform, cfg,
                            n_bootstrap=20, seed=0, before_penalty=None, after_penalty=None):
    blocks = np.asarray(target_cache["block_ids"], dtype=np.int64)
    unique = np.unique(blocks)
    if len(unique) < 2 or len(blocks) == 0:
        return []
    rng = np.random.default_rng(seed)
    improvements = []
    by_block = {b: np.flatnonzero(blocks == b) for b in unique}
    for _ in range(int(n_bootstrap)):
        sampled_blocks = rng.choice(unique, size=len(unique), replace=True)
        ids = np.concatenate([by_block[b] for b in sampled_blocks if len(by_block[b])])
        if len(ids) == 0: continue
        s0, _ = _score_embeddings(before_emb[0], before_emb[1], source_y, target_cache, transform, cfg, ids,
                                  residual_penalty_value=before_penalty)
        s1, _ = _score_embeddings(after_emb[0], after_emb[1], source_y, target_cache, transform, cfg, ids,
                                  residual_penalty_value=after_penalty)
        improvements.append(float(s0 - s1))
    return improvements


def _make_transport_source(train_patches, train_labels, num_classes, per_class, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for c in range(num_classes):
        idx = np.flatnonzero(train_labels == c)
        if len(idx) > per_class: idx = rng.choice(idx, size=int(per_class), replace=False)
        selected.extend(idx.tolist())
    selected = np.asarray(sorted(selected), dtype=int)
    return train_patches[selected], train_labels[selected]


def _train_epochs(model, teacher, optimizer, scheduler, transform, train_patches, train_labels,
                  device, training_cfg, epoch_count, seed, use_transport=True):
    if len(train_labels) == 0:
        raise ValueError("Source training split is empty; cannot train HFCST")
    ds = TensorDataset(torch.from_numpy(train_patches), torch.from_numpy(train_labels).long())
    batch_size = int(training_cfg.get("batch_size", 16))
    generator = torch.Generator().manual_seed(int(seed))
    if bool(training_cfg.get("class_balanced_sampling", False)):
        labels_np = np.asarray(train_labels, dtype=np.int64)
        counts = np.bincount(labels_np, minlength=int(labels_np.max()) + 1).astype(np.float64)
        class_weights = np.zeros_like(counts)
        present = counts > 0
        class_weights[present] = 1.0 / counts[present]
        weights = torch.as_tensor(class_weights[labels_np], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(labels_np), replacement=True, generator=generator
        )
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False,
                            drop_last=False, num_workers=0, generator=generator)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            drop_last=False, num_workers=0, generator=generator)
    teacher_decay = float(training_cfg.get("ema_decay", 0.99))
    lambda_transport = float(training_cfg.get("lambda_transport", 0.5)) if use_transport else 0.0
    history = []
    for ep in range(int(epoch_count)):
        model.train(); losses = []; ls = []; lt = []
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32); yb = yb.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            source_loss = _supervised_loss(out, yb, training_cfg)
            total = source_loss
            transformed_loss = source_loss.detach() * 0.0
            if use_transport and lambda_transport > 0:
                xt = transform(xb, labels=yb)
                out_t = model(xt)
                transformed_loss = _supervised_loss(out_t, yb, training_cfg)
                total = source_loss + lambda_transport * transformed_loss
            if not torch.isfinite(total):
                raise FloatingPointError("Non-finite HFCST training loss")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg.get("grad_clip", 5.0)))
            optimizer.step(); _ema_update(teacher, model, teacher_decay)
            losses.append(float(total.detach().item())); ls.append(float(source_loss.detach().item()))
            lt.append(float(transformed_loss.detach().item()))
        if scheduler is not None: scheduler.step()
        history.append({"epoch": ep + 1, "loss": float(np.mean(losses)),
                        "source_loss": float(np.mean(ls)), "transformed_loss": float(np.mean(lt))})
    return history


def _risk_and_spectral(model, transform, cal_x, cal_y, device, before_logits, cfg):
    after_logits = source_logits(model, transform, cal_x, cal_y, device)
    risk = source_risk_from_logits(before_logits, after_logits, cal_y, model.num_classes,
                                   margin_tolerance=float(cfg.get("margin_tolerance", 0.05)))
    if len(cal_x):
        diag = transform.diagnostics(_tensor(cal_x, device),
                                     labels=torch.as_tensor(cal_y, dtype=torch.long, device=device))
    else:
        diag = {"gain_min": float("nan"), "gain_max": float("nan"), "offset_min": float("nan"),
                "offset_max": float("nan"), "gain_tv": float("nan"), "offset_tv": float("nan"),
                "relative_spectral_change": float("nan")}
    ok, reasons = spectral_plausibility(diag, cfg)
    return risk, diag, ok, reasons, after_logits


def _candidate_search(model, transform, source_x, source_y, target_cache, target_patches,
                      target_q, constraint_patches, constraint_labels, cal_x, cal_y,
                      device, cfg, param_kind, direction, current_score, before_emb,
                      cycle_idx, log_path, phase):
    eta_candidates = [float(x) for x in cfg.get("eta_candidates", [1.0, 0.5, 0.25, 0.125, 0.0625])]
    if len(cal_x) == 0 or len(target_patches) == 0:
        return False, {"reason": "calibration_or_target_validation_empty", "candidates": []}, current_score
    old_shared = transform.phi_shared.detach().clone()
    old_residual = transform.delta_phi.detach().clone()
    active = transform.active_classes[:, None].to(old_residual.dtype)
    before_penalty = float((float(cfg.get("lambda_residual", 0.01)) * (old_residual * active).square().sum()).item())
    before_logits = source_logits(model, transform, cal_x, cal_y, device)
    candidates_log = []
    best_score = current_score
    for eta in eta_candidates:
        if param_kind == "shared":
            transform.set_parameters(shared=old_shared + eta * torch.as_tensor(direction, device=device))
            transform.set_parameters(residual=old_residual)
        else:
            transform.set_parameters(shared=old_shared, residual=old_residual + eta * torch.as_tensor(direction, device=device).reshape_as(old_residual))
        after_emb = _source_embeddings(transform, model, source_x, source_y)
        after_score, score_diag = _score_embeddings(after_emb[0], after_emb[1],
                                                     torch.as_tensor(source_y, dtype=torch.long, device=device),
                                                     target_cache, transform, cfg)
        boot = _bootstrap_improvements(before_emb, after_emb,
                                       torch.as_tensor(source_y, dtype=torch.long, device=device),
                                       target_cache, transform, cfg,
                                       n_bootstrap=int(cfg.get("bootstrap_replicates", 20)),
                                       seed=int(cfg.get("seed", 1237)) + cycle_idx * 101 + int(eta * 1000),
                                       before_penalty=before_penalty,
                                       after_penalty=float(_residual_penalty(transform, cfg).detach().item()))
        risk, spec_diag, spec_ok, spec_reasons, after_logits = _risk_and_spectral(
            model, transform, cal_x, cal_y, device, before_logits, cfg)
        # The transform maps source to target. Before student training it does not alter
        # target predictions; this is recorded as zero transport-induced pseudo-label flips.
        pseudo_flip = 0.0
        block_stability = _block_assignment_stability(target_q, target_cache["block_ids"])
        acceptance = candidate_acceptance(current_score, after_score, boot, risk, spec_ok,
                                          pseudo_flip, cfg, target_block_stability=block_stability)
        record = {"phase": phase, "cycle": cycle_idx, "eta": eta,
                  "qp_status": "direction_supplied", "direction_norm": float(np.linalg.norm(direction)),
                  "score_diag": score_diag, "bootstrap_improvements": boot,
                  "source_risk": risk, "spectral_diag": spec_diag,
                  "spectral_reasons": spec_reasons, "target_block_stability": block_stability,
                  "acceptance": acceptance}
        candidates_log.append(record); _append_jsonl(log_path, {"event": "candidate", **record})
        print(f"[HFCST][cycle={cycle_idx}][{phase}] eta={eta:g} "
              f"D={current_score:.6g}->{after_score:.6g} accepted={acceptance['accepted']} "
              f"reasons={acceptance['reasons']}", flush=True)
        if acceptance["accepted"]:
            best_score = after_score
            return True, {"candidates": candidates_log, "accepted_eta": eta,
                          "accepted_score": after_score, "accepted_diag": record}, best_score
    transform.set_parameters(shared=old_shared, residual=old_residual)
    return False, {"candidates": candidates_log, "reason": "all_candidates_rejected"}, current_score


def _qp_direction(model, transform, source_x, source_y, target_cache, constraint_patches,
                  constraint_labels, device, cfg, parameter_kind="shared"):
    idx, y, competitors = select_constraint_samples(
        model, constraint_patches, constraint_labels, device,
        per_class=int(cfg.get("per_class_constraint_samples", 12)),
        stable_fraction=float(cfg.get("stable_fraction", 0.25)))
    if len(idx) == 0:
        return {"status": "no_constraints_rejected", "reason": "no_correct_constraint_samples",
                "d": np.zeros(transform.dim if parameter_kind == "shared" else transform.num_classes * transform.dim,
                               dtype=np.float32), "d_norm": 0.0, "solver": "none", "num_constraints": 0}
    G, margins, n_bad, valid_constraint_rows = compute_margin_gradients(
        model, transform, constraint_patches[idx.numpy()], y.numpy(), competitors.numpy(), device,
        parameter_kind=parameter_kind)
    source_xt = _tensor(source_x, device)
    source_yt = torch.as_tensor(source_y, dtype=torch.long, device=device)
    objective, objective_diag = _transport_tensor(transform, model, source_xt, source_yt, target_cache, cfg)
    param = transform.phi_shared if parameter_kind == "shared" else transform.delta_phi
    grad = torch.autograd.grad(objective, param, retain_graph=False, create_graph=False, allow_unused=True)[0]
    if grad is None: grad = torch.zeros_like(param)
    grad_np = grad.detach().cpu().numpy().reshape(-1).astype(np.float64)
    solver_cfg = cfg
    if parameter_kind == "shared":
        qp = solve_margin_qp(grad_np, G, margins, rho_m=float(solver_cfg.get("rho_m", 0.05)),
                             lambda_d=float(solver_cfg.get("lambda_d", 0.01)),
                             trust_radius=float(solver_cfg.get("shared_trust_radius", 0.15)))
    else:
        # Residuals are solved independently per class so each class has its own trust region.
        C, D = transform.num_classes, transform.dim
        d_all = np.zeros((C, D), dtype=np.float32)
        statuses = {}
        max_constraint_residual = 0.0
        # G/margins have non-finite rows removed by compute_margin_gradients;
        # use the same row map for labels or class-specific constraints can be misassigned.
        y_np = y.numpy()[valid_constraint_rows]
        for c in range(C):
            if not bool(transform.active_classes[c].item()):
                statuses[str(c)] = "unsupported_class_fixed_zero"; continue
            rows = np.flatnonzero(y_np == c)
            if len(rows) == 0:
                statuses[str(c)] = "no_class_margin_constraints"; continue
            gc = G[rows, c*D:(c+1)*D]
            qp_c = solve_margin_qp(grad_np.reshape(C, D)[c], gc, margins[rows],
                                   rho_m=float(solver_cfg.get("rho_m", 0.05)),
                                   lambda_d=float(solver_cfg.get("lambda_d", 0.01)),
                                   trust_radius=float(solver_cfg.get("residual_trust_radius", 0.05)))
            statuses[str(c)] = {"status": qp_c["status"], "solver": qp_c["solver"],
                                "d_norm": qp_c["d_norm"],
                                "constraint_residual_max": qp_c.get("constraint_residual_max")}
            max_constraint_residual = max(max_constraint_residual, float(qp_c.get("constraint_residual_max", 0.0)))
            if qp_c["status"] == "optimal": d_all[c] = qp_c["d"]
        qp = {"status": "optimal" if np.linalg.norm(d_all) > 1e-8 else "no_residual_direction",
              "solver": "per_class_qp", "d": d_all.reshape(-1), "d_norm": float(np.linalg.norm(d_all)),
              "directional_derivative": float(np.dot(grad_np, d_all.reshape(-1))),
              "constraint_residual_max": max_constraint_residual, "num_constraints": int(len(margins)),
              "per_class_status": statuses, "reason": "ok" if np.linalg.norm(d_all) > 1e-8 else "zero_residual_direction"}
    qp["objective_before"] = float(objective.detach().cpu().item())
    qp["objective_diag"] = objective_diag
    qp["invalid_gradient_rows"] = int(n_bad)
    qp["margin_gradient_shape"] = list(G.shape)
    return qp


def _bootstrap_paired_difference(before_emb, after_emb, source_y, before_cache, after_cache,
                                 transform, cfg, n_bootstrap=20, seed=0):
    """Spatial-block paired bootstrap of pre/post-training transport score difference."""
    blocks = np.asarray(before_cache["block_ids"], dtype=np.int64)
    unique = np.unique(blocks)
    if len(unique) < 2 or len(blocks) == 0 or len(after_cache["block_ids"]) != len(blocks):
        return []
    rng = np.random.default_rng(seed)
    by_block = {b: np.flatnonzero(blocks == b) for b in unique}
    diffs = []
    penalty = float(_residual_penalty(transform, cfg).detach().item())
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        ids = np.concatenate([by_block[b] for b in sampled if len(by_block[b])])
        if len(ids) == 0: continue
        s0, _ = _score_embeddings(before_emb[0], before_emb[1], source_y,
                                  before_cache, transform, cfg, ids, residual_penalty_value=penalty)
        s1, _ = _score_embeddings(after_emb[0], after_emb[1], source_y,
                                  after_cache, transform, cfg, ids, residual_penalty_value=penalty)
        diffs.append(float(s0 - s1))
    return diffs


def _posterior_check(model, teacher, transform, source_x, source_y, target_patches,
                     target_cache_before, target_q, cal_x, cal_y, device, cfg, cycle_idx, log_path):
    # q/r are the frozen pre-candidate validation assignments; only embeddings are recomputed.
    post_cache = _target_cache(model, target_patches, target_q["q"], target_q["r"],
                               target_cache_before["block_ids"].numpy() if torch.is_tensor(target_cache_before["block_ids"])
                               else target_cache_before["block_ids"], device)
    post_cache["q"] = target_cache_before["q"].clone()
    post_cache["r"] = target_cache_before["r"].clone()
    pre_emb = _source_embeddings(transform, model, source_x, source_y)
    post_score, post_diag = _score_embeddings(pre_emb[0], pre_emb[1],
                                               torch.as_tensor(source_y, dtype=torch.long, device=device),
                                               post_cache, transform, cfg)
    # Post-training target predictions are checked against the frozen teacher assignment.
    target_logits = post_cache["logits"]
    qpred = target_cache_before["q"].argmax(1)
    student_pred = target_logits.argmax(1).numpy() if len(target_logits) else np.empty(0, dtype=np.int64)
    target_flip = float(np.mean(student_pred != qpred.numpy())) if len(qpred) else 1.0
    block_flip = _block_flip_rates(qpred.numpy(), student_pred, target_cache_before["block_ids"])
    # Compare source calibration before/after training under the same accepted transform.
    # pre-training logits are passed by caller through cfg temporary cache.
    pre_logits = cfg.pop("_pretrain_cal_logits", None)
    pre_emb = cfg.pop("_pretrain_embeddings", None)
    post_emb = _source_embeddings(transform, model, source_x, source_y)
    post_logits = source_logits(model, transform, cal_x, cal_y, device)
    if pre_logits is None:
        risk = {"available": False, "reason": "missing_pretraining_calibration_snapshot"}
    else:
        risk = source_risk_from_logits(pre_logits, post_logits, cal_y, model.num_classes,
                                       margin_tolerance=float(cfg.get("margin_tolerance", 0.05)))
    spec_diag = transform.diagnostics(_tensor(cal_x, device),
                                      labels=torch.as_tensor(cal_y, dtype=torch.long, device=device)) if len(cal_x) else {}
    spec_ok, spec_reasons = spectral_plausibility(spec_diag, cfg) if spec_diag else (False, ["empty_calibration"])
    pre_score = float(cfg.pop("_pretrain_transport_score", float("nan")))
    before_cache = cfg.pop("_pretrain_target_cache", None)
    tolerance = float(cfg.get("post_transport_tolerance", 0.02))
    boot = (_bootstrap_paired_difference(
        pre_emb, post_emb, torch.as_tensor(source_y, dtype=torch.long, device=device),
        before_cache, post_cache, transform, cfg,
        n_bootstrap=int(cfg.get("bootstrap_replicates", 20)),
        seed=int(cfg.get("seed", 1237)) + cycle_idx * 211 + 17)
        if pre_emb is not None and before_cache is not None else [])
    boot = np.asarray(boot, dtype=float)
    boot = boot[np.isfinite(boot)]
    boot_ci_low = (float(np.quantile(boot, 1.0 - float(cfg.get("bootstrap_confidence", 0.90))))
                   if len(boot) else None)
    reasons = []
    if not np.isfinite(post_score) or post_score > pre_score + tolerance:
        reasons.append("post_training_transport_noninferiority_failed")
    if len(boot) < int(cfg.get("minimum_bootstrap_replicates", 5)):
        reasons.append("post_training_bootstrap_unavailable")
    elif boot_ci_low is not None and boot_ci_low < -tolerance:
        reasons.append("post_training_bootstrap_transport_degradation")
    if not risk.get("available", False): reasons.append("post_training_source_risk_unavailable")
    else:
        if not risk.get("class_coverage_complete", True):
            reasons.append("post_training_source_class_coverage_incomplete")
        if risk["new_error_rate"] > float(cfg.get("max_new_error_rate", 0.02)):
            reasons.append("post_training_source_error_budget_exceeded")
        if risk["worst_class_new_error"] > float(cfg.get("max_class_new_error_rate", 0.10)):
            reasons.append("post_training_class_error_budget_exceeded")
        if risk["margin_loss"] > float(cfg.get("max_margin_loss", 0.20)):
            reasons.append("post_training_margin_budget_exceeded")
    if not spec_ok: reasons.append("post_training_spectral_plausibility_failed")
    if target_flip > float(cfg.get("max_target_pseudo_flip_rate", 0.25)):
        reasons.append("post_training_target_pseudo_flip_budget_exceeded")
    if (block_flip.get("available", False) and
            block_flip["max_block_flip_rate"] > float(cfg.get("max_block_target_flip_rate", 0.35))):
        reasons.append("post_training_block_target_pseudo_flip_budget_exceeded")
    result = {"accepted": not reasons, "reasons": reasons, "pre_transport_score": pre_score,
              "post_transport_score": post_score, "post_transport_bootstrap_differences": boot.tolist(),
              "post_transport_bootstrap_ci_low": boot_ci_low, "post_transport_diag": post_diag,
              "post_source_risk": risk, "spectral_diag": spec_diag,
              "spectral_reasons": spec_reasons, "target_pseudo_flip_rate": target_flip,
              "target_block_flip_rates": block_flip}
    _append_jsonl(log_path, {"event": "posterior_acceptance", "cycle": cycle_idx, **result})
    return result


def capture_transaction_state(model, optimizer, teacher, transform, scheduler, committed_cycles=0):
    """Deep snapshot of every mutable state participating in an adaptation transaction."""
    return {"model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "teacher": copy.deepcopy(teacher.state_dict()),
            "transform": copy.deepcopy(transform.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
            "committed_cycles": int(committed_cycles)}


def restore_transaction_state(snapshot, model, optimizer, teacher, transform, scheduler):
    """Restore a full transaction snapshot, including optimizer momentum and EMA buffers."""
    model.load_state_dict(snapshot["model"])
    optimizer.load_state_dict(snapshot["optimizer"])
    teacher.load_state_dict(snapshot["teacher"])
    transform.load_state_dict(snapshot["transform"])
    if scheduler is not None and snapshot.get("scheduler") is not None:
        scheduler.load_state_dict(snapshot["scheduler"])
    return int(snapshot.get("committed_cycles", 0))


def _save_checkpoint(path, model, transform, metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "transform": transform.state_dict(),
                "bands": model.bands, "num_classes": model.num_classes,
                "feature_dim": model.feature_dim, "max_gate": model.max_gate,
                "metadata": _jsonable(metadata)}, path)


def _target_metrics(model, patches, labels, device, num_classes):
    logits = _eval_logits(model, patches, device)
    y = torch.as_tensor(labels, dtype=torch.long)
    pred = logits.argmax(1)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for a, b in zip(y.tolist(), pred.tolist()):
        if 0 <= a < num_classes and 0 <= b < num_classes: cm[a, b] += 1
    total = int(cm.sum().item())
    oa = float(cm.diag().sum().item() / max(1, total))
    cls_acc = []
    f1s = []
    for c in range(num_classes):
        tp = int(cm[c, c]); fn = int(cm[c, :].sum() - tp); fp = int(cm[:, c].sum() - tp)
        if tp + fn > 0: cls_acc.append(tp / (tp + fn))
        denom = 2*tp + fp + fn
        if denom > 0: f1s.append(2*tp/denom)
    aa = float(np.mean(cls_acc)) if cls_acc else None
    macro_f1 = float(np.mean(f1s)) if f1s else None
    row = cm.sum(1).double(); col = cm.sum(0).double()
    pe = float((row * col).sum().item() / max(1, total*total))
    kappa = float((oa - pe) / (1.0 - pe)) if abs(1-pe) > 1e-12 else 0.0
    return {"n": total, "OA": oa, "AA": aa, "MacroF1": macro_f1, "Kappa": kappa,
            "confusion_matrix": cm.tolist(), "per_class_accuracy": cls_acc}


def run_hfcst(cfg: dict, device: torch.device, config_path=None):
    root = Path(__file__).resolve().parents[1]
    cfg = copy.deepcopy(cfg)
    data_cfg, tr_cfg, tc_cfg = cfg["dataset"], cfg.get("training", {}), cfg.get("hfcst", {})
    out_cfg = cfg.get("output", {})
    seed = int(out_cfg.get("seed", tr_cfg.get("seed", 1237)))
    _seed_all(seed)
    num_classes = int(data_cfg["num_classes"])
    patch_size = int(tr_cfg.get("patch_size", 11))

    # Optional restoration from a saved HFCST checkpoint.
    restore_ckpt = None
    restore_ckpt_path = None
    restore_meta = {}
    restore_path = tr_cfg.get("init_checkpoint")
    if restore_path:
        restore_ckpt_path = _resolve_path(restore_path, root)
        if not restore_ckpt_path.is_file():
            raise FileNotFoundError(
                f"Initialization checkpoint not found: {restore_ckpt_path}"
            )
        restore_ckpt = torch.load(restore_ckpt_path, map_location="cpu")
        restore_meta = restore_ckpt.get("metadata", {})
        if not isinstance(restore_meta, dict):
            raise ValueError("Invalid checkpoint metadata")
        if "shared_source_scale" not in restore_meta:
            raise ValueError("Checkpoint has no shared_source_scale")
        if "model" not in restore_ckpt or "transform" not in restore_ckpt:
            raise ValueError("Checkpoint must contain model and transform states")
        if int(restore_ckpt.get("num_classes", -1)) != num_classes:
            raise ValueError("Checkpoint num_classes does not match config")
        ckpt_patch_size = int(restore_meta.get("patch_size", patch_size))
        if patch_size != ckpt_patch_size:
            raise ValueError(
                f"patch_size mismatch: config={patch_size}, "
                f"checkpoint={ckpt_patch_size}"
            )
        print(
            f"[HFCST] restoring checkpoint: {restore_ckpt_path}",
            flush=True,
        )
    grid_size = int(tr_cfg.get("spatial_grid_size", 8))
    output_dir = _resolve_path(out_cfg.get("save_dir", "results_hfcst/default"), root)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (restore_ckpt_path is not None
            and output_dir.resolve() == restore_ckpt_path.parent.resolve()):
        raise ValueError(
            "Use a new output.save_dir; do not write into the checkpoint directory"
        )
    log_path = output_dir / "events.jsonl"
    if log_path.exists(): log_path.unlink()
    print(f"[HFCST] task={data_cfg.get('source_name')}->{data_cfg.get('target_name')} device={device} output={output_dir}", flush=True)

    src_path = _resolve_path(data_cfg["source_image"], root)
    src_gt_path = _resolve_path(data_cfg["source_gt"], root)
    tgt_path = _resolve_path(data_cfg["target_image"], root)
    # Crucial: target GT is intentionally not opened anywhere in this pre-evaluation path.
    src_cube, src_valid, src_audit = load_cube(src_path, data_cfg.get("expected_bands", {}).get("source"))
    tgt_cube, tgt_valid, tgt_audit = load_cube(tgt_path, data_cfg.get("expected_bands", {}).get("target"))
    src_gt, src_gt_key = load_gt(src_gt_path, expected_shape=src_cube.shape[:2])
    src_cube, tgt_cube, band_audit = align_cubes(src_cube, tgt_cube, data_cfg.get("band_mapping"))
    if src_cube.shape[:2] != src_gt.shape:
        raise ValueError("Source image dimensions changed unexpectedly after band alignment")
    # Split source globally before scale estimation. The target GT path remains untouched.
    split, split_blocks, split_audit = global_spatial_split(
        src_gt, src_valid, num_classes, patch_size=patch_size, grid_size=grid_size, seed=seed,
        train_cap=int(tr_cfg.get("per_class_train_cap", 150)),
        constraint_cap=int(tr_cfg.get("per_class_constraint_cap", tc_cfg.get("per_class_constraint_samples", 12))),
        calibration_cap=int(tr_cfg.get("per_class_calibration_cap", 80)),
        validation_cap=int(tr_cfg.get("per_class_validation_cap", 150)),
        block_overrides=tr_cfg.get("block_split_overrides"))
    overlap_audit = audit_no_cross_split_patch_overlap(split, patch_size)
    if not overlap_audit["no_cross_split_overlap"]:
        raise RuntimeError(f"Patch support overlap detected: {overlap_audit}")
    if bool(tr_cfg.get("strict_source_class_coverage", True)):
        missing = split_audit.get("missing_classes_by_split", {})
        incomplete = {name: missing.get(name, []) for name in ("train", "validation")
                      if missing.get(name, [])}
        if incomplete:
            raise RuntimeError(
                "Spatial source split has missing classes in train/validation: "
                f"{incomplete}. Full per-split class counts: {split_audit.get('class_counts', {})}. "
                "Do not train/select a checkpoint on incomplete class coverage; "
                "adjust training.block_split_overrides or set strict_source_class_coverage=false "
                "only if this limitation is intentional."
            )
    if len(split["train"]) == 0:
        raise RuntimeError("No source training samples after strict spatial isolation; increase grid size or check source labels")
    if restore_ckpt is not None:
        checkpoint_bands = int(restore_ckpt.get("bands", -1))
        if (checkpoint_bands != int(src_cube.shape[-1])
                or checkpoint_bands != int(tgt_cube.shape[-1])):
            raise ValueError(
                f"Checkpoint expects {checkpoint_bands} bands, but aligned "
                f"source/target have {src_cube.shape[-1]}/{tgt_cube.shape[-1]}"
            )
        scale = float(restore_meta["shared_source_scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid checkpoint scale: {scale}")
        scale_audit = {
            "quantile": float(restore_meta.get(
                "source_scale_quantile", tr_cfg.get("scale_quantile", 0.95)
            )),
            "mode": "restored_checkpoint",
            "checkpoint": str(restore_ckpt_path),
        }
        print(f"[HFCST] restored source scale={scale}", flush=True)
    else:
        scale, scale_audit = robust_source_scale(
            src_cube, split["train"],
            float(tr_cfg.get("scale_quantile", 0.95))
        )
    src_cube = src_cube / scale
    tgt_cube = tgt_cube / scale  # shared scale
    src_patches = {s: build_patches(src_cube, xy, patch_size) for s, xy in split.items()}
    src_labels = {s: _labels_at(src_gt, xy) for s, xy in split.items()}

    tgt_split, tgt_blocks, tgt_split_audit = global_target_split(
        tgt_valid, patch_size=patch_size, grid_size=grid_size, seed=seed + 31,
        estimate_ratio=float(tr_cfg.get("target_estimate_ratio", 0.70)),
        max_estimate=int(tr_cfg.get("target_pool_size", 512)),
        max_validation=int(tr_cfg.get("target_validation_pool_size", 384)))
    tgt_patches = {s: build_patches(tgt_cube, xy, patch_size) for s, xy in tgt_split.items()}
    audit = {"source": src_audit, "target": tgt_audit, "source_gt_key": src_gt_key,
             "band_alignment": band_audit, "shared_source_scale": scale_audit,
             "source_split": split_audit, "source_patch_overlap": overlap_audit,
             "target_split": tgt_split_audit,
             "source_split_sample_counts": {s: int(len(x)) for s, x in split.items()},
             "source_split_class_counts_zero_based": {s: {str(c): int((src_labels[s] == c).sum()) for c in range(num_classes)} for s in split},
             "target_gt_loaded_before_final_checkpoint": False,
             "target_soft_assignments": "teacher q and reliability r; no target GT"}
    (output_dir / "data_audit.json").write_text(json.dumps(_jsonable(audit), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[data] source split={audit['source_split_sample_counts']} target split={tgt_split_audit}", flush=True)

    feature_dim = (
        int(restore_ckpt.get("feature_dim", tr_cfg.get("feature_dim", 64)))
        if restore_ckpt is not None else int(tr_cfg.get("feature_dim", 64))
    )
    max_gate = (
        float(restore_ckpt.get("max_gate", tr_cfg.get("max_gate", 0.5)))
        if restore_ckpt is not None else float(tr_cfg.get("max_gate", 0.5))
    )
    model = HFCSTModel(
        src_cube.shape[-1], num_classes,
        feature_dim=feature_dim, max_gate=max_gate
    ).to(device)
    transform = HierarchicalSpectralTransform(
        src_cube.shape[-1], num_classes, K=int(tc_cfg.get("K", 8)),
        rho_a=float(tc_cfg.get("rho_a", 0.10)),
        rho_b=float(tc_cfg.get("rho_b", 0.10))
    ).to(device)

    if restore_ckpt is not None:
        model.load_state_dict(restore_ckpt["model"], strict=True)
        transform.load_state_dict(restore_ckpt["transform"], strict=True)
        print(
            "[HFCST] model and spectral transform restored strictly",
            flush=True,
        )

    teacher = copy.deepcopy(model).to(device).eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    train_x, train_y = src_patches["train"], src_labels["train"]
    constraint_x, constraint_y = src_patches["constraint"], src_labels["constraint"]
    cal_x, cal_y = src_patches["calibration"], src_labels["calibration"]
    val_x, val_y = src_patches["validation"], src_labels["validation"]
    if len(train_y) == 0: raise RuntimeError("Empty source training split")
    warm_epochs = 0 if restore_ckpt is not None else int(tr_cfg.get("epochs_warmup", 20))
    total_epochs = warm_epochs + int(tc_cfg.get("num_cycles", 2)) * int(tr_cfg.get("epochs_per_cycle", 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tr_cfg.get("lr", 3e-4)),
                                  weight_decay=float(tr_cfg.get("weight_decay", 4e-4)))
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs), eta_min=1e-6)
                 if tr_cfg.get("lr_scheduler", "cosine") == "cosine" else None)

    # Source-only warm-up and source-only checkpoint selection.
    best_state = None; best_val_score = float("-inf"); warm_history = []
    for ep in range(warm_epochs):
        hist = _train_epochs(model, teacher, optimizer, scheduler, transform, train_x, train_y,
                             device, tr_cfg, 1, seed + ep, use_transport=False)
        val_metrics = _source_val_metrics(model, val_x, val_y, device, num_classes)
        row = {**hist[0], "epoch": ep + 1, "source_validation": val_metrics}
        warm_history.append(row); _append_jsonl(log_path, {"event": "warmup_epoch", **row})
        if val_metrics["available"]:
            score = 0.5 * float(val_metrics["oa"]) + 0.5 * float(val_metrics["aa"])
            if score > best_val_score:
                best_val_score = score
                best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
        teacher.load_state_dict(best_state)
        # Reset optimizer/scheduler so the selected baseline and optimizer state agree.
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(tr_cfg.get("lr", 3e-4)),
                                      weight_decay=float(tr_cfg.get("weight_decay", 4e-4)))
        adaptation_epochs = max(1, int(tc_cfg.get("num_cycles", 2)) * int(tr_cfg.get("epochs_per_cycle", 2)))
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=adaptation_epochs, eta_min=1e-6)
                     if tr_cfg.get("lr_scheduler", "cosine") == "cosine" else None)
    baseline_meta = {
        "stage": (
            "restored_checkpoint_baseline"
            if restore_ckpt is not None else "source_only_baseline"
        ),
        "initialized_from_checkpoint": (
            str(restore_ckpt_path) if restore_ckpt_path is not None else None
        ),
        "source_validation": _source_val_metrics(
            model, val_x, val_y, device, num_classes
        ),
        "warmup_epochs": warm_epochs,
        "config_path": str(config_path) if config_path else None,
        "seed": seed,
        "shared_source_scale": scale,
        "patch_size": patch_size,
        "band_alignment": band_audit,
        "source_scale_quantile": scale_audit["quantile"],
    }
    _save_checkpoint(output_dir / "baseline.pt", model, transform, baseline_meta)
    _append_jsonl(log_path, {"event": "baseline_saved", **baseline_meta})
    print(f"[HFCST] baseline saved; source validation={baseline_meta['source_validation']}", flush=True)

    # Build fixed, balanced source samples used only to estimate/validate transport direction.
    trans_x_np, trans_y_np = _make_transport_source(
        train_x, train_y, num_classes, int(tc_cfg.get("transport_source_per_class", 24)), seed + 777)
    trans_x = _tensor(trans_x_np, device)
    trans_y = torch.as_tensor(trans_y_np, dtype=torch.long, device=device)
    rollback_count = 0; committed_cycles = 0; adaptation_stopped_reason = None
    for cycle in range(int(tc_cfg.get("num_cycles", 2))):
        # Full transaction snapshot: model, optimizer, scheduler, teacher, transform and counters.
        snapshot = capture_transaction_state(model, optimizer, teacher, transform, scheduler, committed_cycles)
        q_est = _teacher_assignments(teacher, tgt_patches["estimate"], device, num_classes,
                                     temperature=float(tc_cfg.get("teacher_temperature", 1.0)),
                                     perturb_std=float(tc_cfg.get("spectral_perturb_std", 0.01)),
                                     seed=seed + cycle * 17 + 1)
        q_val = _teacher_assignments(teacher, tgt_patches["validation"], device, num_classes,
                                     temperature=float(tc_cfg.get("teacher_temperature", 1.0)),
                                     perturb_std=float(tc_cfg.get("spectral_perturb_std", 0.01)),
                                     seed=seed + cycle * 17 + 2)
        est_cache = _target_cache(model, tgt_patches["estimate"], q_est["q"], q_est["r"],
                                  tgt_blocks["estimate"], device)
        val_cache = _target_cache(model, tgt_patches["validation"], q_val["q"], q_val["r"],
                                  tgt_blocks["validation"], device)
        eff_est = [effective_sample_size(torch.as_tensor(q_est["r"] * q_est["q"][:, c])) for c in range(num_classes)]
        eff_val = [effective_sample_size(torch.as_tensor(q_val["r"] * q_val["q"][:, c])) for c in range(num_classes)]
        source_counts = [int((trans_y == c).sum().item()) for c in range(num_classes)]
        supported = [eff_est[c] >= float(tc_cfg.get("min_effective_target_n", 8.0)) and
                     eff_val[c] >= float(tc_cfg.get("min_effective_target_n", 8.0)) and
                     source_counts[c] >= int(tc_cfg.get("min_source_per_class", 4)) for c in range(num_classes)]
        transform.set_supported_classes(supported)
        if len(tgt_patches["estimate"]) == 0 or len(tgt_patches["validation"]) == 0:
            cycle_diag = {"cycle": cycle, "accepted": False, "reason": "target_estimate_or_validation_unavailable",
                          "effective_target_n_estimate": eff_est, "effective_target_n_validation": eff_val,
                          "supported_classes": supported}
            _append_jsonl(log_path, {"event": "cycle_rejected", **cycle_diag})
            adaptation_stopped_reason = cycle_diag["reason"]
            transform.load_state_dict(snapshot["transform"])
            break
        before_emb = _source_embeddings(transform, model, trans_x, trans_y)
        before_score, before_diag = _score_embeddings(before_emb[0], before_emb[1], trans_y,
                                                       val_cache, transform, {**tc_cfg, "num_classes": num_classes})
        objective, objective_diag = _transport_tensor(transform, model, trans_x, trans_y, est_cache,
                                                       {**tc_cfg, "num_classes": num_classes})
        _append_jsonl(log_path, {"event": "cycle_start", "cycle": cycle, "transport_objective_estimate": float(objective.detach().item()),
                                 "transport_validation_before": before_score, "transport_validation_diag": before_diag,
                                 "effective_target_n_estimate": eff_est, "effective_target_n_validation": eff_val,
                                 "supported_classes": supported, "target_pred_distribution_estimate": q_est["pred_dist"],
                                 "target_pred_distribution_validation": q_val["pred_dist"],
                                 "target_block_stability_estimate": _block_assignment_stability(q_est, tgt_blocks["estimate"]),
                                 "target_block_stability_validation": _block_assignment_stability(q_val, tgt_blocks["validation"]),
                                 "mean_reliability_estimate": float(np.mean(q_est["r"])) if len(q_est["r"]) else None,
                                 "mean_reliability_validation": float(np.mean(q_val["r"])) if len(q_val["r"]) else None,
                                 "objective_diag": objective_diag})
        if not any(supported):
            transform.load_state_dict(snapshot["transform"])
            adaptation_stopped_reason = "no_reliable_target_class_support"
            _append_jsonl(log_path, {"event": "cycle_rejected", "cycle": cycle, "reason": adaptation_stopped_reason,
                                     "pure_source_ce_executed": False})
            break
        # Shared QP first.
        qp = _qp_direction(model, transform, trans_x, trans_y, est_cache, constraint_x,
                           constraint_y, device, {**tc_cfg, "num_classes": num_classes}, "shared")
        _append_jsonl(log_path, {"event": "qp", "cycle": cycle, "phase": "shared", **qp})
        if qp.get("status") != "optimal" or qp.get("d_norm", 0.0) < float(tc_cfg.get("min_direction_norm", 1e-7)):
            transform.load_state_dict(snapshot["transform"])
            adaptation_stopped_reason = f"shared_qp_rejected:{qp.get('status')}"
            _append_jsonl(log_path, {"event": "cycle_rejected", "cycle": cycle, "reason": adaptation_stopped_reason,
                                     "pure_source_ce_executed": False})
            break
        shared_ok, shared_diag, shared_score = _candidate_search(
            model, transform, trans_x, trans_y, val_cache, tgt_patches["validation"], q_val,
            constraint_x, constraint_y, cal_x, cal_y, device,
            {**tc_cfg, "num_classes": num_classes, "seed": seed}, "shared", qp["d"], before_score,
            before_emb, cycle, log_path, "shared")
        if not shared_ok:
            transform.load_state_dict(snapshot["transform"])
            adaptation_stopped_reason = "shared_candidate_all_rejected"
            _append_jsonl(log_path, {"event": "cycle_rejected", "cycle": cycle, "reason": adaptation_stopped_reason,
                                     "pure_source_ce_executed": False, "shared_diagnostics": shared_diag})
            break

        # Optional class residual QPs only after shared candidate was accepted.
        shared_residual = transform.delta_phi.detach().clone()
        residual_diag = {"attempted": False, "accepted": False, "reason": "disabled_or_zero_direction"}
        if bool(tc_cfg.get("enable_class_residuals", True)) and any(supported):
            residual_qp = _qp_direction(model, transform, trans_x, trans_y, est_cache, constraint_x,
                                        constraint_y, device, {**tc_cfg, "num_classes": num_classes}, "residual")
            _append_jsonl(log_path, {"event": "qp", "cycle": cycle, "phase": "residual", **residual_qp})
            if residual_qp.get("status") == "optimal" and residual_qp.get("d_norm", 0.0) >= float(tc_cfg.get("min_direction_norm", 1e-7)):
                shared_emb = _source_embeddings(transform, model, trans_x, trans_y)
                shared_score, _ = _score_embeddings(shared_emb[0], shared_emb[1], trans_y, val_cache, transform,
                                                     {**tc_cfg, "num_classes": num_classes})
                residual_ok, residual_diag, residual_score = _candidate_search(
                    model, transform, trans_x, trans_y, val_cache, tgt_patches["validation"], q_val,
                    constraint_x, constraint_y, cal_x, cal_y, device,
                    {**tc_cfg, "num_classes": num_classes, "seed": seed}, "residual", residual_qp["d"],
                    shared_score, shared_emb, cycle, log_path, "class_residual")
                residual_diag["attempted"] = True; residual_diag["accepted"] = bool(residual_ok)
                if not residual_ok:
                    transform.set_parameters(residual=shared_residual)
            else:
                residual_diag = {"attempted": True, "accepted": False,
                                 "reason": residual_qp.get("reason", residual_qp.get("status")),
                                 "qp": residual_qp}
        # Candidate is now pre-validated. Train only under an accepted transform.
        pretrain_cal_logits = source_logits(model, transform, cal_x, cal_y, device)
        pretrain_emb = _source_embeddings(transform, model, trans_x, trans_y)
        pretrain_score, pretrain_score_diag = _score_embeddings(pretrain_emb[0], pretrain_emb[1], trans_y,
                                                                 val_cache, transform, {**tc_cfg, "num_classes": num_classes})
        tc_runtime = {**tc_cfg, "num_classes": num_classes, "seed": seed,
                      "_pretrain_cal_logits": pretrain_cal_logits,
                      "_pretrain_embeddings": pretrain_emb,
                      "_pretrain_target_cache": val_cache,
                      "_pretrain_transport_score": pretrain_score}
        cycle_history = _train_epochs(model, teacher, optimizer, scheduler, transform,
                                      train_x, train_y, device, {**tr_cfg, **tc_cfg},
                                      int(tr_cfg.get("epochs_per_cycle", 2)), seed + 1000 + cycle,
                                      use_transport=True)
        posterior = _posterior_check(model, teacher, transform, trans_x, trans_y,
                                     tgt_patches["validation"], val_cache, q_val, cal_x, cal_y,
                                     device, tc_runtime, cycle, log_path)
        if not posterior["accepted"]:
            committed_cycles = restore_transaction_state(snapshot, model, optimizer, teacher, transform, scheduler)
            rollback_count += 1
            adaptation_stopped_reason = "post_training_acceptance_failed"
            _append_jsonl(log_path, {"event": "transaction_rollback", "cycle": cycle,
                                     "rollback_count": rollback_count, "restored_model_optimizer_teacher_transform_scheduler": True,
                                     "posterior": posterior, "pure_source_ce_after_rejection": False})
            print(f"[HFCST][cycle={cycle}] posterior rejected; full rollback and stop", flush=True)
            break
        committed_cycles += 1
        cycle_diag = {"cycle": cycle, "accepted": True, "shared": shared_diag,
                      "residual": residual_diag, "pretrain_transport_score": pretrain_score,
                      "pretrain_transport_diag": pretrain_score_diag, "training_history": cycle_history,
                      "posterior": posterior, "committed_cycles": committed_cycles,
                      "rollback_count": rollback_count}
        _append_jsonl(log_path, {"event": "cycle_committed", **cycle_diag})
        _save_checkpoint(output_dir / "last_committed.pt", model, transform,
                         {"stage": "committed_adaptation", "cycle": cycle, "committed_cycles": committed_cycles,
                          "seed": seed, "config_path": str(config_path) if config_path else None,
                          "shared_source_scale": scale, "patch_size": patch_size})
        print(f"[HFCST][cycle={cycle}] committed; score={posterior['post_transport_score']:.6g}", flush=True)

    # Final model is selected solely from source-only warm-up plus transactionally accepted cycles.
    final_meta = {"stage": "final_before_target_evaluation", "committed_cycles": committed_cycles,
                  "rollback_count": rollback_count, "adaptation_stopped_reason": adaptation_stopped_reason,
                  "seed": seed, "target_gt_used_for_selection": False,
                  "shared_source_scale": scale, "patch_size": patch_size,
                  "band_alignment": band_audit, "source_scale_quantile": scale_audit["quantile"]}
    final_path = output_dir / "final.pt"
    _save_checkpoint(final_path, model, transform, final_meta)
    _append_jsonl(log_path, {"event": "final_checkpoint_saved", **final_meta})
    print(f"[HFCST] final checkpoint saved: {final_path}; committed_cycles={committed_cycles}", flush=True)

    # Only now may target GT be opened, and it is used only for final reporting.
    final_metrics = None
    target_gt_path = _resolve_path(data_cfg["target_gt"], root) if data_cfg.get("target_gt") else None
    if target_gt_path is not None and target_gt_path.exists():
        target_gt, target_gt_key = load_gt(target_gt_path, expected_shape=tgt_cube.shape[:2])
        valid_eval = tgt_valid & (target_gt > 0) & (target_gt <= num_classes)
        coords_eval = np.argwhere(valid_eval)
        eval_patches = build_patches(tgt_cube, coords_eval, patch_size)
        eval_labels = _labels_at(target_gt, coords_eval)
        final_metrics = _target_metrics(model, eval_patches, eval_labels, device, num_classes)
        final_metrics.update({"gt_key": target_gt_key, "gt_loaded_after_final_checkpoint": True,
                              "selection_used_target_gt": False})
        print(f"[FINAL TARGET EVAL - not used for selection] {final_metrics}", flush=True)
    else:
        final_metrics = {"available": False, "reason": "target_gt_file_missing_or_not_configured",
                         "gt_loaded_after_final_checkpoint": False, "selection_used_target_gt": False}
        print("[FINAL TARGET EVAL] skipped: target GT file unavailable", flush=True)
    results = {"task": f"{data_cfg.get('source_name')}->{data_cfg.get('target_name')}",
               "config": str(config_path) if config_path else None, "seed": seed,
               "device": str(device), "baseline_source_validation": baseline_meta["source_validation"],
               "committed_cycles": committed_cycles, "rollback_count": rollback_count,
               "adaptation_stopped_reason": adaptation_stopped_reason,
               "target_metrics_final_only": final_metrics, "target_gt_used_for_selection": False,
               "final_checkpoint": str(final_path), "data_audit": str(output_dir / "data_audit.json"),
               "event_log": str(log_path)}
    (output_dir / "results.json").write_text(json.dumps(_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8")
    _append_jsonl(log_path, {"event": "run_complete", **results})
    return results
