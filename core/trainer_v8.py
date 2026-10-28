"""T-CST v8.0: strict compliance with two design docs.
   - Backbone: DualReliabilityEncoder (doc principle 3)
   - Domain gradient: raw (doc 5.3, no unit-normalization)
   - Eta candidates: starting at 1.0 (doc 8.2)
   - Only doc 8.7 accept rules (no extra target_flip / phi_max_norm)
   - Checkpoint selected by source val only (doc 13.1, no oracle)
   - Target GT loaded ONLY after training ends (principle 4)
   - Target transport / calibration from disjoint subsets (doc 8.1)
   - Train 30 warmup + 6 cycles x 5 (doc section 10)
   - 10% source training labels (user requirement)
"""
from __future__ import annotations

from contextlib import contextmanager as _cm

@_cm
def _freeze_bn(model):
    """T8: Safe 分支 forward 时冻结 BN 统计（eval 模式），梯度仍通过 BN 仿射参数。"""
    bn_modules = [m for m in model.modules()
                  if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    states = [m.training for m in bn_modules]
    try:
        for m in bn_modules:
            m.eval()
        yield
    finally:
        for m, st in zip(bn_modules, states):
            m.train(st)


import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from core.data import (align_bands, build_patches, load_scene,
                       sample_unlabeled, _load_mat, _pick_2d)
from core.discrepancy import (domain_gradient_mmd, spectral_descriptor,
                              target_consensus_statistics,
                              build_spectral_shape_centroids,
                              spectral_shape_posteriors,
                              fuse_class_posteriors)
from core.metrics import classification_metrics, confusion_matrix
from core.model import DualReliabilityEncoder
from core.pseudo_sampling import (sample_pseudo_indices,
                                   select_anchored_rescue_indices)
from core.risk_controller import backtracking_search
from core.spectral_transform import SmoothSpectralTransform, ClassSmoothSpectralTransform
from core.task_geometry import compute_margin_gradients, select_constraint_samples
from core.transport_solver import solve_margin_qp


CLASS_NAMES = {
    "houston": ["grass_healthy", "grass_stressed", "trees", "water",
                "residential", "non_residential", "road"],
    "hyrank":  [f"class{i}" for i in range(1, 13)],
    "hangzhou_shanghai": ["class1", "class2", "class3"],
}


def _log(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def _make_balanced_loader(patches, labels, class_weights, batch_size, seed):
    """Doc 5.2 class-balanced principle extended to CE training."""
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    sw = class_weights[labels].astype(np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sw),
        num_samples=len(labels),
        replacement=True,
        generator=torch.Generator().manual_seed(int(seed)))
    ds = TensorDataset(torch.from_numpy(patches), torch.from_numpy(labels))
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      drop_last=False, num_workers=0)


def _make_model(bands, num_classes, device, cfg=None):
    """Doc principle 3: DualReliabilityEncoder only. Reads architecture from cfg['training']."""
    tr = (cfg or {}).get("training", {})
    proj_dim = int(tr.get("projection_dim", 128))
    proj_norm = tr.get("projection_norm", "batch")
    aspp_mode = tr.get("aspp_mode", "adaptive")
    pool_mode = tr.get("pooling_mode", "guided")
    fusion_mode = tr.get("spectral_fusion_mode", "reliable")
    amp_ch = bool(tr.get("amplitude_channel", False))
    _log(f"[backbone] DualReliabilityEncoder (bands={bands}, classes={num_classes}, "
         f"proj_dim={proj_dim}, proj_norm={proj_norm}, aspp={aspp_mode}, "
         f"pool={pool_mode}, fusion={fusion_mode}, amp={amp_ch})")
    return DualReliabilityEncoder(
        bands=bands, num_classes=num_classes, projection_dim=proj_dim,
        projection_norm=proj_norm, aspp_mode=aspp_mode, pooling_mode=pool_mode,
        spectral_fusion_mode=fusion_mode, amplitude_channel=amp_ch).to(device)


def _build_scheduler(optimizer, name, total_epochs):
    if name in (None, "none", "None"):
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs), eta_min=1e-6)
    raise ValueError(f"Unknown lr_scheduler: {name}")


def _eval_source(model, patches, labels, device, use_cb_val=False,
                 ckpt_mode="A", num_classes=None, target_patches=None,
                 return_details=False):
    """
    ckpt_mode:
      "A" = accuracy (original), higher is better
      "B" = accuracy + target marginal entropy tie-breaker, higher is better
      "C" = class-balanced negative NLL, higher is better
    Returns a single float score (higher is always better).
    """
    model.eval()
    preds = []
    all_logits = []
    with torch.no_grad():
        for s in range(0, len(patches), 256):
            xb = torch.from_numpy(patches[s:s+256]).to(device)
            lg = model(xb).logits.float().cpu()
            all_logits.append(lg)
            preds.append(lg.argmax(1).numpy())
    preds = np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)
    logits = torch.cat(all_logits, 0) if all_logits else torch.empty(0)
    y_true = np.asarray(labels)

    # 基础 accuracy
    if not use_cb_val:
        acc = 100.0 * float((preds == y_true).sum()) / max(1, len(patches))
    else:
        classes = np.unique(y_true)
        accs = []
        for c in classes:
            mask = (y_true == c)
            if mask.sum() > 0:
                accs.append((preds[mask] == c).sum() / mask.sum())
        acc = 100.0 * float(np.mean(accs))

    # nan 保护：如果 acc 是 nan（空 val 或 accs 为空），返回 0.0
    if acc is None or (isinstance(acc, float) and np.isnan(acc)):
        acc = 0.0

    # P1.2: 总是计算 class-balanced NLL（不管 ckpt_mode，manifest 需要）
    if num_classes is None:
        num_classes = int(y_true.max()) + 1 if len(y_true) > 0 else 1
    nll = float('inf')
    try:
        if len(y_true) > 0 and logits.numel() > 0:
            y = torch.from_numpy(y_true).long()
            log_probs = torch.log_softmax(logits, dim=1)
            per_nll = -log_probs.gather(1, y[:, None]).squeeze(1).numpy()
            class_nlls = []
            for c in range(num_classes):
                m = (y_true == c)
                if m.sum() > 0:
                    class_nlls.append(per_nll[m].mean())
            nll = float(np.mean(class_nlls)) if class_nlls else 0.0
    except Exception:
        nll = float('inf')

    if ckpt_mode == "A":
        score = acc
    elif ckpt_mode == "C":
        score = -nll if nll != float('inf') else -1e9
    elif ckpt_mode == "D":
        # P1.2: D 模式内层只返回 acc，最终选择在外层做（合格集 + NLL 排序）
        score = acc
    elif ckpt_mode == "B":
        if target_patches is None or len(target_patches) == 0 or num_classes is None:
            score = acc
        else:
            with torch.no_grad():
                tp = []
                for s in range(0, len(target_patches), 256):
                    xb = torch.from_numpy(target_patches[s:s+256]).to(device)
                    tp.append(model(xb).logits.argmax(1).cpu().numpy())
                tp = np.concatenate(tp) if tp else np.empty(0, dtype=np.int64)
            if len(tp) == 0:
                score = acc
            else:
                counts = np.bincount(tp, minlength=num_classes).astype(np.float64)
                p = counts / max(1.0, counts.sum())
                H = -float(np.sum(p * np.log(p + 1e-8)))
                H_norm = H / np.log(max(2, num_classes))
                score = acc + 0.1 * H_norm
    else:
        score = acc

    if return_details:
        return score, {"acc": float(acc), "nll": float(nll)}
    return score

def _eval_target(model, patches, labels, device, num_classes, use_tta=False):
    model.eval(); preds = []
    with torch.no_grad():
        for s in range(0, len(patches), 256):
            xb = torch.from_numpy(patches[s:s+256]).to(device)
            if use_tta:
                logits = model(xb).logits
                logits = logits + model(torch.flip(xb, dims=(-1,))).logits
                logits = logits + model(torch.flip(xb, dims=(-2,))).logits
                logits = logits / 3.0
                preds.append(logits.argmax(1).cpu().numpy())
            else:
                preds.append(model(xb).logits.argmax(1).cpu().numpy())
    pred = np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)
    return classification_metrics(confusion_matrix(labels, pred, num_classes))


def _split_source_doc(gt, train_ratio, constraint_per_class, calibration_per_class,
                      val_frac_from_train, seed, cal_pool_mult=3, per_class_train_cap=0, per_class_val_cap=0, per_class_val_target=0):
    """Doc 8.1: disjoint train / constraint / cal_pool / val.
       BA-CS: keep a larger cal_pool (3x); final calibration set is
       boundary-aware and chosen AFTER warm-up based on source margins.
    """
    rng = np.random.default_rng(seed)
    masks = {k: np.zeros_like(gt, dtype=bool)
             for k in ("train", "constraint", "cal_pool", "val")}
    stats = {"train": 0, "constraint": 0, "cal_pool": 0, "val": 0}
    val_frac_from_train = float(val_frac_from_train)
    for c in [int(i) for i in np.unique(gt) if i > 0]:
        coords = np.argwhere(gt == c)
        rng.shuffle(coords)
        n = len(coords)
        n_cn = min(constraint_per_class, max(1, n // 10))
        remaining = n - n_cn
        if train_ratio >= 1.0 - 1e-6:
            if per_class_train_cap > 0:
                n_tr = min(remaining, int(per_class_train_cap))
            else:
                n_tr = remaining
            n_val_total = remaining - n_tr
            if per_class_val_target > 0:
                n_val_total = min(n_val_total, int(per_class_val_target))
            elif per_class_val_cap > 0:
                n_val_total = min(n_val_total, int(per_class_val_cap))
        else:
            n_tr = max(1, int(round(n * train_ratio)))

            if per_class_train_cap > 0:

                n_tr = min(n_tr, int(per_class_train_cap))
            n_tr = min(n_tr, max(1, remaining - 1))
            n_val_total = remaining - n_tr
            if per_class_val_cap > 0:
                n_val_total = min(n_val_total, int(per_class_val_cap))
        n_ca_pool = min(calibration_per_class * int(cal_pool_mult), max(1, n_val_total // 2))
        n_val = max(0, n_val_total - n_ca_pool)
        slices = {
            "train": coords[:n_tr],
            "constraint": coords[n_tr:n_tr+n_cn],
            "cal_pool": coords[n_tr+n_cn:n_tr+n_cn+n_ca_pool],
            "val": coords[n_tr+n_cn+n_ca_pool:n_tr+n_cn+n_ca_pool+n_val],
        }
        total = sum(len(v) for v in slices.values())
        assert total <= n, f"split overflow for class {c}: {total} > {n}"
        for k, arr in slices.items():
            if len(arr):
                masks[k][arr[:, 0], arr[:, 1]] = True
                stats[k] += len(arr)
    return masks, stats

def _spatial_block_split(gt, train_ratio, constraint_per_class, calibration_per_class,
                         val_frac_from_train, seed, cal_pool_mult=3,
                         per_class_train_cap=0, per_class_val_cap=0,
                         per_class_val_target=0, grid_size=10, patch_size=11,
                         strict_mixed_isolation=False):
    """空间分块划分：train 和 val 来自不同块，避免 11×11 patch 重叠。

    对于标签数 < 400 的类，允许混合划分（1 个块给 train，剩余给 val），
    但会明确记录"混合类"，供论文声明。
    """
    rng = np.random.default_rng(seed)
    H, W = gt.shape
    block_h = max(1, H // grid_size)
    block_w = max(1, W // grid_size)
    pixel_block = np.full((H, W), -1, dtype=np.int64)
    for by in range(grid_size):
        for bx in range(grid_size):
            r0 = by * block_h
            r1 = (by + 1) * block_h if by < grid_size - 1 else H
            c0 = bx * block_w
            c1 = (bx + 1) * block_w if bx < grid_size - 1 else W
            pixel_block[r0:r1, c0:c1] = by * grid_size + bx

    masks = {k: np.zeros_like(gt, dtype=bool)
             for k in ("train", "constraint", "cal_pool", "val")}
    stats = {"train": 0, "constraint": 0, "cal_pool": 0, "val": 0}
    mixed_classes = []

    for c in [int(i) for i in np.unique(gt) if i > 0]:
        coords = np.argwhere(gt == c)
        blocks = pixel_block[coords[:, 0], coords[:, 1]]
        unique_blocks = np.unique(blocks)
        n_c = len(coords)
        target_train = per_class_train_cap if per_class_train_cap > 0 else int(round(n_c * train_ratio))
        target_val = per_class_val_target if per_class_val_target > 0 else max(1, n_c // 5)

        if len(unique_blocks) < 2 or n_c < 2 * (target_train + target_val):
            # 混合模式：允许 train 和 val 共享块，但中心不同
            mixed_classes.append(c)
            rng.shuffle(coords)
            n_cn = min(constraint_per_class, max(1, n_c // 10))
            n_ca_pool = min(calibration_per_class * int(cal_pool_mult),
                           max(1, (n_c - n_cn) // 3))
            remaining = n_c - n_cn - n_ca_pool
            # 修复 Bug：按比例分配，防止 train 抢光导致 val 只有 1 个
            if remaining >= (target_train + target_val):
                n_tr = target_train
                n_val = target_val
            else:
                n_tr = min(target_train, max(1, int(remaining * 0.5)))
                n_val = max(1, remaining - n_tr)
            cn_coords = coords[:n_cn]
            ca_coords = coords[n_cn:n_cn + n_ca_pool]
            tr_coords = coords[n_cn + n_ca_pool:n_cn + n_ca_pool + n_tr]
            # A1 fix: exclude val coords within patch_size of any train coord
            _ps = int(patch_size)
            _cand = coords[n_cn + n_ca_pool + n_tr:n_cn + n_ca_pool + n_tr + n_val + 100]
            if len(tr_coords) > 0 and len(_cand) > 0:
                _tr_set = tr_coords
                _keep = []
                for _v in _cand:
                    _d = np.abs(_tr_set - _v).max(axis=1)  # Chebyshev distance
                    if (_d >= _ps).all():
                        _keep.append(_v)
                    if len(_keep) >= n_val:
                        break
                # B5 fix: 严格隔离不可取消。如果隔离后样本 <5，扩大候选池重试；
                # 仍然不足时，保留隔离后的全部样本（哪怕只有 3-4 个），并标记。
                if len(_keep) >= max(5, n_val // 2):
                    va_coords = np.array(_keep, dtype=np.int64)
                else:
                    # 扩大候选池（多取 200 个），再隔离一次
                    _cand_wide = coords[n_cn + n_ca_pool + n_tr:
                                         n_cn + n_ca_pool + n_tr + n_val + 200]
                    _keep_wide = []
                    for _v in _cand_wide:
                        _d = np.abs(_tr_set - _v).max(axis=1)
                        if (_d >= _ps).all():
                            _keep_wide.append(_v)
                        if len(_keep_wide) >= n_val:
                            break
                    if len(_keep_wide) >= 1:
                        va_coords = np.array(_keep_wide, dtype=np.int64)
                    else:
                        # B5b 兜底：真的找不到隔离样本 → 退回未隔离样本（有重叠但至少有 val）
                        # 这比返回空（导致 _eval_source 返回 nan）更安全
                        if strict_mixed_isolation:
                            # P0.3: strict 模式禁止 fallback 到未隔离，隔离后剩多少用多少
                            va_coords = np.empty((0, 2), dtype=np.int64)
                            _log(f"  [split] class {c} mixed: STRICT, 0 isolated val (fallback disabled)")
                        else:
                            va_coords = _cand[:n_val] if len(_cand) > 0 else np.empty((0, 2), dtype=np.int64)
            else:
                if strict_mixed_isolation:
                    va_coords = np.empty((0, 2), dtype=np.int64)
                    _log(f"  [split] class {c} mixed: STRICT, 0 isolated val (no candidate)")
                else:
                    va_coords = _cand[:n_val]
        else:
            # 纯块划分
            rng.shuffle(unique_blocks)
            n_val_blocks = max(1, len(unique_blocks) // 5)
            val_block_set = set(unique_blocks[:n_val_blocks].tolist())
            val_list, train_list = [], []
            for i, b in enumerate(blocks):
                if int(b) in val_block_set:
                    val_list.append(coords[i])
                else:
                    train_list.append(coords[i])
            val_coords_all = np.asarray(val_list, dtype=np.int64)
            train_coords_all = np.asarray(train_list, dtype=np.int64)
            rng.shuffle(train_coords_all)
            n_avail = len(train_coords_all)
            n_cn = min(constraint_per_class, max(1, n_avail // 10))
            n_ca_pool = min(calibration_per_class, max(1, (n_avail - n_cn) // 3))
            remaining = n_avail - n_cn - n_ca_pool
            n_tr = min(target_train, max(1, remaining - 1))
            cn_coords = train_coords_all[:n_cn]
            ca_coords = train_coords_all[n_cn:n_cn + n_ca_pool]
            tr_coords = train_coords_all[n_cn + n_ca_pool:n_cn + n_ca_pool + n_tr]
            # B6 fix: 纯块划分也要跨块边界隔离（Patch 可能跨块边界）
            _ps_b6 = int(patch_size)
            _tr_set_b6 = tr_coords
            _val_keep = []
            for _v in val_coords_all:
                if len(_tr_set_b6) > 0:
                    _d = np.abs(_tr_set_b6 - _v).max(axis=1)
                    if (_d >= _ps_b6).all():
                        _val_keep.append(_v)
                else:
                    _val_keep.append(_v)
                if len(_val_keep) >= target_val:
                    break
            va_coords = (np.asarray(_val_keep, dtype=np.int64)
                         if len(_val_keep) > 0 else val_coords_all[:target_val])

        for arr, key in [(tr_coords, "train"), (cn_coords, "constraint"),
                         (ca_coords, "cal_pool"), (va_coords, "val")]:
            if len(arr):
                masks[key][arr[:, 0], arr[:, 1]] = True
                stats[key] += len(arr)

    return masks, stats, mixed_classes




@torch.no_grad()
def _sample_boundary_calibration(model, pool_patches, pool_labels, per_class, device, seed, enabled=True):
    """BA-CS: pick per-class 50% boundary (low margin) + 50% easy (high margin)."""
    if not enabled:
        if pool_patches is None or len(pool_patches) == 0:
            return None, None, None
        counts = {}
        for c in np.unique(pool_labels):
            counts[int(c)] = int((pool_labels == c).sum())
        return pool_patches, pool_labels, counts
    if pool_patches is None or len(pool_patches) == 0:
        return None, None, None
    model.eval()
    logits_list = []
    for s in range(0, len(pool_patches), 128):
        xb = torch.from_numpy(pool_patches[s:s+128]).to(device)
        logits_list.append(model(xb).logits.float())
    logits = torch.cat(logits_list, 0).cpu()
    y = torch.from_numpy(pool_labels).long()
    other = logits.clone(); other.scatter_(1, y[:, None], -1e9)
    true_logit = logits.gather(1, y[:, None]).squeeze(1)
    margin = (true_logit - other.max(1).values).numpy()

    rng = np.random.default_rng(seed)
    selected = []
    per_class_counts = {}
    for c in np.unique(pool_labels):
        c_idx = np.where(pool_labels == c)[0]
        if len(c_idx) == 0:
            continue
        n_take = min(int(per_class), len(c_idx))
        order = np.argsort(margin[c_idx])
        n_half = n_take // 2
        pick = []
        if n_half > 0:
            pick.extend(c_idx[order[:n_half]].tolist())
            pick.extend(c_idx[order[-n_half:]].tolist())
        if len(pick) < n_take:
            used = set(pick)
            remaining = [i for i in c_idx.tolist() if i not in used]
            if remaining:
                need = min(n_take - len(pick), len(remaining))
                pick.extend(rng.choice(remaining, size=need, replace=False).tolist())
        pick = pick[:n_take]
        selected.extend(pick)
        per_class_counts[int(c)] = len(pick)
    if not selected:
        return None, None, None
    selected = np.asarray(sorted(selected), dtype=np.int64)
    return pool_patches[selected], pool_labels[selected], per_class_counts


def _class_balanced_subsample(labels, total, seed):
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per = max(1, total // len(classes))
    idx = []
    for c in classes:
        ci = np.where(labels == c)[0]
        idx.extend((ci if len(ci) <= per else rng.choice(ci, per, replace=False)).tolist())
    idx = np.asarray(idx, dtype=np.int64)
    rng.shuffle(idx)
    return idx


def _load_gt_map(gt_path):
    d = _load_mat(Path(gt_path))
    arr, _ = _pick_2d(d)
    if arr is None:
        raise KeyError(f"{gt_path}: no 2-D array found")
    return np.asarray(arr, dtype=np.int64)


def _print_per_class(metrics, num_classes, dataset_name):
    names = CLASS_NAMES.get(dataset_name, [f"class{i}" for i in range(num_classes)])
    _log("\nPer-class accuracy:")
    for c in range(num_classes):
        nm = names[c] if c < len(names) else f"class{c}"
        acc = metrics["class_accuracy"][c] if c < len(metrics["class_accuracy"]) else 0.0
        _log(f"  [{c}] {nm:<22s} : {acc:>6.2f}%")


def run_tcst_v8(cfg, device):
    torch.manual_seed(cfg["output"]["seed"]); np.random.seed(cfg["output"]["seed"])
    run_mode = cfg["training"].get("run_mode", "full")
    ds_name = cfg["dataset"]["name"]
    ckpt_mode = cfg["training"].get("ckpt_mode", "A")  # A/B/C
    _log(f"[ckpt] mode={ckpt_mode} (A=accuracy, B=acc+target_entropy, C=class_balanced_negNLL)")

    _log(f"\n{'='*70}\nDataset: {ds_name}  run_mode={run_mode}\n{'='*70}")
    _log(f"[protocol] STRICT DOC COMPLIANCE")
    _log(f"  - g_D: raw (no normalization)")
    _log(f"  - eta candidates: {cfg['tcst']['eta_candidates']}")
    _log(f"  - checkpoint: src_val only")
    _log(f"  - target GT: loaded only AFTER training")
    # P0.4: 动态计算实际训练样本占比，不再硬编码 "10%"
    _train_ratio_val = float(cfg["training"].get("source_train_ratio", 1.0))
    if _train_ratio_val >= 1.0 - 1e-6:
        _log(f"  - source train ratio: 100% (with per_class_train_cap as upper bound)")
    else:
        _log(f"  - source train ratio: {_train_ratio_val*100:.0f}%")

    # Load source (image + gt)
    _use_zscore = bool(cfg["training"].get("use_zscore", False))
    src_cube, src_gt = load_scene(cfg["dataset"]["source_image"],
                                  cfg["dataset"]["source_gt"],
                                  None, cfg["dataset"]["num_classes"], verbose=True,
                                  use_zscore=_use_zscore)
    # Load target WITHOUT GT (doc principle 4)
    tgt_cube, _ = load_scene(cfg["dataset"]["target_image"],
                             None,
                             None, cfg["dataset"]["num_classes"], verbose=True,
                             use_zscore=_use_zscore)
    src_cube, tgt_cube, bands = align_bands(src_cube, tgt_cube,
                                             cfg["dataset"].get("band_stride_source", 1))
    _log(f"[align] bands={bands} num_classes={cfg['dataset']['num_classes']}")

    num_classes = cfg["dataset"]["num_classes"]
    patch_size = cfg["training"]["patch_size"]
    train_ratio = cfg["training"]["source_train_ratio"]
    cn_per = cfg["training"]["constraint_per_class"]
    ca_per = cfg["training"]["calibration_per_class"]

    val_frac = float(cfg["training"].get("val_frac_from_train", 0.10))
    cal_mult = int(cfg["training"].get("cal_pool_multiplier", 3))
    use_spatial = bool(cfg["training"].get("use_spatial_block_split", True))
    grid_size = int(cfg["training"].get("spatial_grid_size", 10))
    if use_spatial:
        split, stats, mixed_classes = _spatial_block_split(
            src_gt, train_ratio, cn_per, ca_per, val_frac,
            seed=cfg["output"]["seed"] + 11,
            cal_pool_mult=cal_mult,
            per_class_train_cap=int(cfg["training"].get("per_class_train_cap", 0)),
            per_class_val_cap=int(cfg["training"].get("per_class_val_cap", 0)),
            per_class_val_target=int(cfg["training"].get("per_class_val_target", 0)),
            grid_size=grid_size, patch_size=patch_size,
            strict_mixed_isolation=bool(cfg["training"].get("strict_mixed_isolation", False)))
        _log(f"[spatial-split] grid={grid_size}x{grid_size}  mixed_classes={mixed_classes}")
    else:
        split, stats = _split_source_doc(src_gt, train_ratio, cn_per, ca_per,
                                          val_frac,
                                          seed=cfg["output"]["seed"] + 11,
                                          cal_pool_mult=cal_mult,
                                          per_class_train_cap=int(cfg["training"].get("per_class_train_cap", 0)),
                                          per_class_val_cap=int(cfg["training"].get("per_class_val_cap", 0)),
                                          per_class_val_target=int(cfg["training"].get("per_class_val_target", 0)))
        _log(f"[spatial-split] DISABLED (random pixel split)")
    def _cl(mask):
        c = np.argwhere(mask)
        return c, (src_gt[c[:, 0], c[:, 1]] - 1).astype(np.int64)

    src_tr_c, src_tr_y = _cl(split["train"])
    src_cn_c, src_cn_y = _cl(split["constraint"])
    src_ca_c, src_ca_y = _cl(split["cal_pool"])   # BA-CS: candidate pool
    src_val_c, src_val_y = _cl(split["val"])
    mode_label = "100%-protocol" if train_ratio >= 1.0 - 1e-6 else f"{train_ratio*100:.0f}%-protocol"
    _log(f"\n[source-split] {mode_label}")
    _log(f"  train={stats['train']}  constraint={stats['constraint']}  "
         f"cal_pool={stats['cal_pool']}  val={stats['val']}")
    # P0.4: 明确 per_class_train_cap 是上限，并打印每类实际训练数 vs 该类总标注数
    _cap = int(cfg["training"].get("per_class_train_cap", 0))
    _tr_counts = np.bincount(src_tr_y, minlength=num_classes).tolist()
    _all_counts = [int((src_gt == c).sum()) for c in range(1, num_classes + 1)]
    _log(f"  per-class train counts (cap={_cap}, upper bound): {_tr_counts}")
    _log(f"  per-class total labeled (before split): {_all_counts}")
    _short = [c for c in range(num_classes) if _tr_counts[c] < _cap]
    if _short:
        _log(f"  note: classes {_short} have train < cap because total labeled pixels are insufficient")

    src_tr_p = build_patches(src_cube, src_tr_c, patch_size)
    src_cn_p = build_patches(src_cube, src_cn_c, patch_size)
    src_ca_p = build_patches(src_cube, src_ca_c, patch_size)
    src_val_p = build_patches(src_cube, src_val_c, patch_size)

    # 稳妥的背景过滤：只过滤纯黑/无效像素（使用极小的绝对阈值，避免误杀水体）
    tgt_energy = np.linalg.norm(tgt_cube, axis=-1)
    energy_threshold = 1e-4  # 只过滤纯黑或接近纯黑的死像素
    tgt_valid_mask = tgt_energy > energy_threshold
    _log(f"[target-mask] absolute_energy_threshold={energy_threshold:.4f}  valid={tgt_valid_mask.sum()}")
    tgt_pool_c = sample_unlabeled(tgt_cube, tgt_valid_mask,
                                  cfg["training"].get("target_pool_size", 4000),
                                  seed=cfg["output"]["seed"] + 17)
    n_half = len(tgt_pool_c) // 2
    tgt_transport_p = build_patches(tgt_cube, tgt_pool_c[:n_half], patch_size)
    tgt_calibration_p = build_patches(tgt_cube, tgt_pool_c[n_half:], patch_size)
    _log(f"[target] pool={len(tgt_pool_c)} "
         f"(transport={n_half} / calibration={len(tgt_pool_c)-n_half}, disjoint)")

    use_rs_crw = bool(cfg["tcst"].get("use_rs_crw", False))
    use_pseudo = bool(cfg["training"].get("use_pseudo", False))
    pseudo_state = {"xb": None, "yb": None, "weights": None, "class_indices": None}
    pseudo_rng = np.random.default_rng(int(cfg["output"]["seed"]) + 3901)
    if use_pseudo:
        _log(f"[Pseudo] enabled, thr={cfg['training'].get('pseudo_threshold', 0.9)}, weight={cfg['training'].get('pseudo_weight', 0.5)}")
    use_ba_cs = bool(cfg["training"].get("use_ba_cs", True))
    rs_crw_alpha = float(cfg["tcst"].get("rs_crw_alpha", 2.0))
    rs_crw_min_cal = int(cfg["tcst"].get("rs_crw_min_cal_samples", 5))
    rs_crw_ema = float(cfg["tcst"].get("rs_crw_ema", 0.5))
    init_weights_from_class_freq = bool(cfg["tcst"].get("init_weights_from_class_freq", False))
    class_risk_weights = np.ones(num_classes, dtype=np.float32)
    if init_weights_from_class_freq:
        _counts = np.bincount(src_tr_y, minlength=num_classes).astype(np.float64)
        _counts = np.maximum(_counts, 1.0)
        _freq_factor = (_counts ** -0.5)
        _freq_factor = _freq_factor / _freq_factor.mean()
        class_risk_weights = _freq_factor.astype(np.float32)
        _log(f"[Opt2] initial class_risk_weights from class freq: "
             f"{[round(float(x),2) for x in class_risk_weights.tolist()]}")
    if use_rs_crw:
        _log(f"\n[RS-CRW] ENABLED: alpha={rs_crw_alpha}  min_cal={rs_crw_min_cal}  ema={rs_crw_ema}")

    model = _make_model(bands, num_classes, device, cfg=cfg)
    use_shape_rescue = bool(cfg["training"].get("spectral_shape_rescue", False))
    shape_centroids = None
    if use_shape_rescue:
        shape_centroids = build_spectral_shape_centroids(
            torch.from_numpy(src_tr_p), torch.from_numpy(src_tr_y), num_classes)
        _log(f"[spectral-shape] enabled, weight={float(cfg['training'].get('spectral_shape_weight', 0.2)):.3f} "
             f"temp={float(cfg['training'].get('spectral_shape_temperature', 0.15)):.3f}")
    previous_target_labels = None

    def _source_prototypes():
        """Build normalized class prototypes from labeled source features only."""
        model.eval()
        dim = int(cfg["training"].get("projection_dim", 128))
        sums = torch.zeros(num_classes, dim, device=device)
        counts = torch.zeros(num_classes, device=device)
        with torch.no_grad():
            for start in range(0, len(src_tr_p), 256):
                xb = torch.from_numpy(src_tr_p[start:start + 256]).to(device)
                yb = torch.from_numpy(src_tr_y[start:start + 256]).long().to(device)
                feat = model(xb).features.float()
                sums.index_add_(0, yb, feat)
                counts.index_add_(0, yb, torch.ones_like(yb, dtype=counts.dtype))
        return F.normalize(sums / counts.clamp_min(1.0).unsqueeze(1), dim=1), counts

    use_prototype_gate = bool(cfg["training"].get("use_prototype_gate", use_pseudo))
    prototype_temperature = float(cfg["training"].get("prototype_temperature", 0.12))
    prototype_margin = float(cfg["training"].get("prototype_margin", 0.03))
    if use_prototype_gate:
        _log(f"[prototype-gate] enabled, temperature={prototype_temperature:.3f}, margin={prototype_margin:.3f}")
    use_class_cond = bool(cfg["tcst"].get("use_class_conditional", False))
    if use_class_cond:
        transform = ClassSmoothSpectralTransform(
            bands, num_classes=num_classes, K=cfg["tcst"]["K"],
            rho_a=cfg["tcst"]["rho_a"], rho_b=cfg["tcst"]["rho_b"]).to(device)
        _log(f"[transform] CLASS-CONDITIONAL: C={num_classes}, K={cfg['tcst']['K']}, dim={transform.dim}")
    else:
        transform = SmoothSpectralTransform(bands, K=cfg["tcst"]["K"],
                                            rho_a=cfg["tcst"]["rho_a"],
                                            rho_b=cfg["tcst"]["rho_b"]).to(device)
        _log(f"[transform] GLOBAL: K={cfg['tcst']['K']}, dim={transform.dim}")
    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"],
                      weight_decay=cfg["training"]["weight_decay"])
    # S1 fix: warmup scheduler 用 warmup 自己的周期（不是总周期）
    # 原逻辑 T_max=total_epochs，cosine 在 warmup 结束时 lr 还没降到 min，停在尖锐局部最优
    # 改成 T_max=warmup_epochs，cosine 在 warmup 结束时 lr 降到 min，收敛到更平坦的解
    total_epochs = cfg["training"]["epochs_warmup"]
    _cycles_total = cfg["training"]["num_cycles"] * cfg["training"]["epochs_per_cycle"]
    _log(f"[scheduler] warmup T_max={total_epochs}, cycles T_max={_cycles_total}")
    scheduler_name = cfg["training"].get("lr_scheduler", "none")
    scheduler = _build_scheduler(optimizer, scheduler_name, total_epochs)
    if scheduler is not None:
        _log(f"[scheduler] {scheduler_name} (T_total={total_epochs})")
    amp = cfg["training"]["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch, "amp") else \
             torch.cuda.amp.GradScaler(enabled=amp)
    save_dir = Path(cfg["output"]["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "training_log.jsonl"

    use_source_aug = bool(cfg["training"].get("use_source_augmentation", False))
    aug_min_class_count = int(cfg["training"].get("aug_min_class_count", 0))
    aug_noise_std = float(cfg["training"].get("aug_noise_std", 0.01))
    aug_scale_range = float(cfg["training"].get("aug_scale_range", 0.08))

    # 类别感知：算出当前训练集中，各类别的样本数
    if use_source_aug and aug_min_class_count > 0:
        counts = np.bincount(src_tr_y, minlength=num_classes)
        aug_class_mask = torch.tensor(
            [counts[c] >= aug_min_class_count for c in range(num_classes)],
            dtype=torch.bool, device=device
        )
        _log(f"[Source-Aug] Class-Aware ENABLED: min_count={aug_min_class_count}, "
             f"augmented_classes={[int(c) for c in range(num_classes) if aug_class_mask[c]]}")
    else:
        aug_class_mask = torch.ones(num_classes, dtype=torch.bool, device=device)
        if use_source_aug:
            _log(f"[Source-Aug] Global ENABLED (all classes)")

    def _apply_source_aug(xb, yb):
        if not use_source_aug:
            return xb
        # T5 fix: 合并 aug_class_mask（按类别样本数保护）和 aug_classes（用户指定）
        base_mask = aug_class_mask[yb]
        _user_aug_classes = cfg["training"].get("aug_classes", None)
        if _user_aug_classes is not None:
            user_allow = torch.zeros_like(base_mask)
            for _c in _user_aug_classes:
                user_allow |= yb.eq(_c)
            base_mask = base_mask & user_allow
        mask = base_mask
        if not mask.any():
            return xb

        B = xb.size(0)
        scale = 1.0 + aug_scale_range * (2 * torch.rand(B, 1, 1, 1, device=xb.device) - 1.0)
        noise = aug_noise_std * torch.randn_like(xb) if aug_noise_std > 0 else 0
        xb_aug = xb * scale + noise
        return torch.where(mask.view(-1, 1, 1, 1), xb_aug, xb)


    label_smoothing_value = float(cfg['training'].get('label_smoothing', 0.035))

    def _focal_loss(logits, targets, gamma=2.0, label_smoothing=0.0):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=label_smoothing)
        pt = torch.exp(-ce)
        return (((1 - pt) ** gamma) * ce).mean()

    def _pseudo_loss(logits, targets, weights, gamma=0.0):
        ce = F.cross_entropy(logits, targets, reduction='none',
                             label_smoothing=label_smoothing_value)
        if gamma > 0:
            ce = (1 - torch.exp(-ce)).pow(gamma) * ce
        w = weights.to(device=logits.device, dtype=ce.dtype)
        return (ce * w).mean()

    def _train_step(patches, labels, phi_star, safe_w):
        model.train(); optimizer.zero_grad(set_to_none=True)
        if isinstance(patches, np.ndarray):
            xb = torch.from_numpy(patches).to(device)
        else:
            xb = patches.to(device, non_blocking=True)
        if isinstance(labels, np.ndarray):
            yb = torch.from_numpy(labels).to(device)
        else:
            yb = labels.to(device, non_blocking=True)
        
        xb_raw = xb
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            focal_gamma = float(cfg["training"].get("focal_gamma", 0.0))
            _ce = (lambda lg, y: _focal_loss(lg, y, gamma=focal_gamma, label_smoothing=label_smoothing_value)) \
                  if focal_gamma > 0 else \
                  (lambda lg, y: F.cross_entropy(lg, y, label_smoothing=label_smoothing_value))
            
            _lambda_spec = float(cfg["training"].get("lambda_spec", 0.0))
            out_clean = model(xb_raw)
            loss_clean = _ce(out_clean.logits, yb)
            if _lambda_spec > 0:
                loss_clean += _lambda_spec * _ce(out_clean.spectral_logits, yb)
            
            _use_safe_training = bool(cfg["training"].get("use_safe_training", True))
            _safe_executed = False
            loss_safe = None
            
            if phi_star is not None and float(phi_star.abs().sum()) > 1e-8:
                _safe_executed = True
                xb_safe = transform(xb_raw, phi_star, labels=yb)
                xb_safe = _apply_source_aug(xb_safe, yb)  # <-- 补回这行！
                with _freeze_bn(model):
                    out_safe = model(xb_safe)
                    loss_safe = _ce(out_safe.logits, yb)
                    if _lambda_spec > 0:
                        loss_safe += _lambda_spec * _ce(out_safe.spectral_logits, yb)
            
            if loss_safe is not None and not _use_safe_training:
                loss = loss_safe
            elif loss_safe is not None:
                loss = (loss_clean + safe_w * loss_safe) / (1 + safe_w)
            else:
                loss = loss_clean
            
            # Pseudo-label loss
            _use_pseudo = bool(cfg["training"].get("use_pseudo", False))
            if _use_pseudo and pseudo_state["xb"] is not None:
                pb_size = xb.size(0)
                idx_p = sample_pseudo_indices(pseudo_state["yb"], pb_size, pseudo_rng)
                p_xb = torch.from_numpy(pseudo_state["xb"][idx_p]).to(device)
                p_yb = torch.from_numpy(pseudo_state["yb"][idx_p]).to(device)
                p_w = torch.from_numpy(pseudo_state["weights"][idx_p]).to(device)
                pseudo_w = float(cfg["training"].get("pseudo_weight", 0.5))
                
                out_pseudo = model(p_xb)
                loss_pseudo = _pseudo_loss(out_pseudo.logits, p_yb, p_w,
                                           gamma=focal_gamma)
                if _lambda_spec > 0:
                    loss_pseudo += _lambda_spec * _pseudo_loss(
                        out_pseudo.spectral_logits, p_yb, p_w, gamma=focal_gamma)
                loss = loss + pseudo_w * loss_pseudo

        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        return {
            "loss": float(loss.detach()),
            "clean_loss": float(loss_clean.detach()),
            "safe_loss": float(loss_safe.detach()) if loss_safe is not None else None,
            "safe_executed": _safe_executed,
            "pseudo_draw_counts": (np.bincount(
                pseudo_state["yb"][idx_p], minlength=num_classes).tolist()
                if _use_pseudo and pseudo_state["xb"] is not None else None),
            "pseudo_repeated_draws": (int(len(idx_p) - len(np.unique(idx_p)))
                                      if _use_pseudo and pseudo_state["xb"] is not None else 0),
        }

    # ------------- warm-up -------------
    # === 优化1: CE class-balanced ===
    cb_power = float(cfg["training"].get("ce_class_balance_power", 0.5))
    counts = np.bincount(src_tr_y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    class_weights_for_ce = (counts.mean() / counts) ** cb_power
    class_weights_for_ce = class_weights_for_ce / class_weights_for_ce.mean()
    _log(f"[class-balanced CE] power={cb_power}  weights={[round(float(w),2) for w in class_weights_for_ce]}")

    _log(f"\n[Phase A] warm-up {cfg['training']['epochs_warmup']} epochs")
    batch = cfg["training"]["batch_size"]
    rng = np.random.default_rng(cfg["output"]["seed"] + 21)
    t0 = time.time()
    best_src_val = -1.0
    best_state = None
    # P1.1: 记录所有 warmup checkpoint 的元数据（不保存 state_dict，只记指标）
    warmup_ckpt_manifest = []
    if cb_power > 0:
        warmup_loader = _make_balanced_loader(src_tr_p, src_tr_y, class_weights_for_ce,
                                              batch, seed=cfg["output"]["seed"] + 21)
        def _iter_warmup():
            for xb_batch, yb_batch in warmup_loader:
                yield xb_batch, yb_batch
    else:
        def _iter_warmup():
            idx = rng.permutation(len(src_tr_p))
            for s in range(0, len(idx), batch):
                sel = idx[s:s+batch]
                yield src_tr_p[sel], src_tr_y[sel]

    for ep in range(1, cfg["training"]["epochs_warmup"] + 1):
        losses = []
        for xb_batch, yb_batch in _iter_warmup():
            losses.append(_train_step(xb_batch, yb_batch, None, 0.0))
        src_val, _warm_details = _eval_source(
            model, src_val_p, src_val_y, device,
            use_cb_val=cfg["training"].get("use_cb_val", False),
            ckpt_mode=ckpt_mode, num_classes=num_classes,
            target_patches=tgt_transport_p[:256] if ckpt_mode == "B" else None,
            return_details=True)
        src_nll = _warm_details["nll"]
        # P1.1: 记录每个 warmup epoch 的指标
        _warm_entry = {
            "epoch": ep, "src_val": float(src_val), "src_nll": float(src_nll),
            "loss_mean": float(np.mean([l["loss"] for l in losses])),
            "clean_loss_mean": float(np.mean([l["clean_loss"] for l in losses])),
            "safe_loss_mean": None,
            "aug_loss_mean": float(np.mean([l["aug_loss"] for l in losses if l.get("aug_loss") is not None])) if any(l.get("aug_loss") is not None for l in losses) else None,
            "safe_executed_count": 0,
            "aug_executed_count": sum(1 for l in losses if l.get("aug_executed", False)),
            "is_best": False,
        }
        if src_val > best_src_val:
            best_src_val = src_val
            best_state = copy.deepcopy(model.state_dict())
            _warm_entry["is_best"] = True
        warmup_ckpt_manifest.append(_warm_entry)
        _aug_count = sum(1 for l in losses if l.get("aug_executed", False))
        _log(f"  warm {ep:03d}/{cfg['training']['epochs_warmup']}  "
             f"loss={np.mean([l['loss'] for l in losses]):.4f}  "
             f"lr={optimizer.param_groups[0]['lr']:.2e}  "
             f"src_val={src_val:.2f}  best={best_src_val:.2f}  "
             f"aug={_aug_count}/{len(losses)}  t={(time.time()-t0)/60:.1f}m")
        if scheduler is not None:
            scheduler.step(src_val) if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else scheduler.step()
    if best_state is not None:
        model.load_state_dict(best_state)
        _log(f"[Phase A] restored best warmup (src_val={best_src_val:.2f})")
        # 重置优化器，避免动量与回退的模型不一致
        optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"],
                          weight_decay=cfg["training"]["weight_decay"])
        # === A2 fix: rebuild scheduler on the new optimizer ===
        if scheduler is not None:
            # T4 fix: T_max = 剩余周期（cycles），不是总周期
            remaining_epochs = cfg["training"]["num_cycles"] * cfg["training"]["epochs_per_cycle"]
            scheduler = _build_scheduler(optimizer, scheduler_name, remaining_epochs)
            _log(f"[scheduler] re-bound: {scheduler_name}, T_max={remaining_epochs} (remaining after warmup)")
        _log("[Optimizer] re-initialized after restoring best warmup model")

    # === Direction B: BA-CS ===
    # T9d: 独立 domain calibration（在 BA-CS 之前定义，随机抽源域训练样本）
    _dom_rng = np.random.default_rng(cfg["output"]["seed"] + 777)
    _dom_idx = _dom_rng.choice(len(src_tr_p), min(128, len(src_tr_p)), replace=False)
    src_domain_cal_p = src_tr_p[_dom_idx]
    ba_cs_ca_p, ba_cs_ca_y, ba_cs_counts = _sample_boundary_calibration(
        model, src_ca_p, src_ca_y,
        per_class=cfg["training"]["calibration_per_class"],
        device=device,
        seed=cfg["output"]["seed"] + 777, enabled=use_ba_cs)
    if ba_cs_ca_p is not None:
        src_ca_p, src_ca_y = ba_cs_ca_p, ba_cs_ca_y
        _log(f"[BA-CS] boundary calibration: {len(src_ca_p)} samples "
             f"per-class={ba_cs_counts}")
    else:
        _log(f"[BA-CS] WARNING: boundary sampling failed, using full cal_pool "
             f"({len(src_ca_p)} samples)")

    if run_mode == "warmup_only" or cfg["training"]["num_cycles"] == 0:
        _log("[WARMUP-ONLY] skip cycles; loading target GT for final eval")
        _manifest_data = {
            "warmup_manifest": warmup_ckpt_manifest,
            "cycle_manifest": [],
            "final_selected_source": "warmup",
            "final_selected_cycle": None,
            "final_selected_epoch": None,
            "final_selected_src_val": float(best_src_val),
            "warmup_best_src_val": float(best_src_val),
            "force_final_epoch": bool(cfg["training"].get("force_final_epoch", False)),
            "ckpt_mode": cfg["training"].get("ckpt_mode", "A"),
            "note": "warmup_only mode: no cycles executed",
        }
        return _final_eval(model, transform, cfg, device, save_dir, best_src_val,
                            stats, [], ds_name, num_classes, bands, tgt_cube, patch_size,
                            manifest_data=_manifest_data,
                            best_state_for_diff=best_state)

    # ------------- alternating cycles -------------
    _log(f"\n[Phase B+C] {cfg['training']['num_cycles']} cycles × "
         f"{cfg['training']['epochs_per_cycle']} epochs")
    cycle_logs = []
    cycle_ckpt_manifest = []
    best_global = {"src_val": best_src_val, "state": copy.deepcopy(model.state_dict()),
                   "source": "warmup", "cycle": None, "epoch": None}
    teacher_model = copy.deepcopy(model).to(device)
    teacher_model.eval()
    for _p in teacher_model.parameters():
        _p.requires_grad_(False)
    teacher_momentum = float(cfg["training"].get("teacher_momentum", 0.99))
    use_teacher_consistency = bool(cfg["training"].get("teacher_consistency", use_pseudo))
    if use_teacher_consistency:
        _log(f"[teacher] EMA enabled, momentum={teacher_momentum:.4f}")

    def _update_teacher():
        """EMA only parameters; copy non-floating buffers exactly."""
        student_state = model.state_dict()
        with torch.no_grad():
            for _name, _t in teacher_model.state_dict().items():
                _s = student_state[_name].detach()
                if _t.is_floating_point():
                    _t.mul_(teacher_momentum).add_(_s, alpha=1.0 - teacher_momentum)
                else:
                    _t.copy_(_s)
        teacher_model.eval()
    # P1.2: D 模式的独立选择（合格条件 + NLL 排序）
    _d_tolerance = float(cfg["training"].get("ckpt_d_tolerance", 2.0))
    _d_base_src_val = float(best_src_val)  # warmup best 作为基准
    _d_best_state = None
    _d_best_nll = float('inf')
    _d_best_meta = None

    for cycle in range(1, cfg["training"]["num_cycles"] + 1):
        _log(f"\n--- Cycle {cycle} ---")
        model.eval()
        qp = None

        # Use all unlabeled transport samples only when pseudo-label training needs them.
        _tgt_rng = np.random.default_rng(cfg["output"]["seed"] + 500 + cycle)
        _n_tgt = min(128, len(tgt_transport_p))
        _tgt_idx = _tgt_rng.choice(len(tgt_transport_p), _n_tgt, replace=False)
        _target_eval_idx = (np.arange(len(tgt_transport_p)) if use_pseudo else _tgt_idx)
        _context_logits, _spectral_logits, _fused_logits, _spectral_reliability = [], [], [], []
        _target_features = []
        _teacher_fused_logits = []
        with torch.no_grad():
            for _s in range(0, len(_target_eval_idx), 128):
                _idx = _target_eval_idx[_s:_s+128]
                _xb = torch.from_numpy(tgt_transport_p[_idx]).to(device)
                _out = model(_xb)
                _teacher_out = teacher_model(_xb)
                _context_logits.append(_out.context_logits.float().cpu())
                _spectral_logits.append(_out.spectral_logits.float().cpu())
                _fused_logits.append(_out.logits.float().cpu())
                _target_features.append(_out.features.float().cpu())
                _spectral_reliability.append(_out.spectral_reliability.float().cpu())
                _teacher_fused_logits.append(_teacher_out.logits.float().cpu())
        _context_all = torch.cat(_context_logits)
        _spectral_all = torch.cat(_spectral_logits)
        _fused_all = torch.cat(_fused_logits)
        _target_features_all = torch.cat(_target_features).numpy()
        _teacher_all = torch.cat(_teacher_fused_logits)
        _teacher_probs = torch.softmax(_teacher_all, dim=1)
        _teacher_conf_t, _teacher_pl_t = _teacher_probs.max(1)
        _student_fused_probs = torch.softmax(_fused_all, dim=1)
        _student_fused_conf_t, _student_fused_pl_t = _student_fused_probs.max(1)
        _teacher_student_agree = _teacher_pl_t.eq(_student_fused_pl_t)
        if use_teacher_consistency and use_pseudo:
            _log(f"  [teacher] student_agreement={float(_teacher_student_agree.float().mean()):.3f} "
                 f"teacher_conf={float(_teacher_conf_t.mean()):.3f}")
        _probs, _target_weights, _target_agreement = target_consensus_statistics(
            _context_all, _spectral_all, _fused_all,
            torch.cat(_spectral_reliability))
        _shape_probs = None
        if use_shape_rescue:
            _shape_probs = spectral_shape_posteriors(
                torch.from_numpy(tgt_transport_p[_target_eval_idx]), shape_centroids,
                temperature=float(cfg["training"].get("spectral_shape_temperature", 0.15)))
            _probs = fuse_class_posteriors(
                _probs, _shape_probs,
                shape_weight=float(cfg["training"].get("spectral_shape_weight", 0.2)))
            _log(f"  [spectral-shape] confidence={float(_shape_probs.max(1).values.mean()):.3f}")
        _log(f"  [target-consensus] agreement={float(_target_agreement.float().mean()):.3f}  "
             f"reliable_weight={float(_target_weights.sum()):.1f}/{len(_target_weights)}")

        if use_pseudo:
            _conf, _pl = _probs.max(1)
            _thr = float(cfg["training"].get("pseudo_threshold", 0.8))
            _conf = _conf.numpy()
            _pl = _pl.numpy()
            _weights = _target_weights.numpy()
            _agreement = _target_agreement.numpy()
            _all_agree = ((_context_all.argmax(1) == _spectral_all.argmax(1))
                          & (_context_all.argmax(1) == _fused_all.argmax(1))).numpy()
            _teacher_conf = _teacher_conf_t.numpy()
            _teacher_pl = _teacher_pl_t.numpy()
            _student_fused_conf = _student_fused_conf_t.numpy()
            _student_fused_pl = _student_fused_pl_t.numpy()
            _teacher_agree = _teacher_student_agree.numpy()
            _rescue_thr = float(cfg["training"].get("pseudo_rescue_threshold", 0.55))
            if use_teacher_consistency:
                _teacher_ok = (_teacher_agree
                               & (_teacher_pl == _student_fused_pl)
                               & (_teacher_conf >= _thr))
                _teacher_rescue_ok = (_teacher_ok & (_teacher_conf >= _rescue_thr))
            else:
                _teacher_ok = np.ones_like(_teacher_agree, dtype=bool)
                _teacher_rescue_ok = _teacher_ok

            # Main pseudo labels require all three views to agree. Two-view samples
            # remain available to MMD, but do not directly train the classifier.
            _per_class = int(cfg["training"].get("pseudo_per_class", 200))
            _rescue_min_main = int(cfg["training"].get("pseudo_rescue_min_main", 32))
            _rescue_per_class = int(cfg["training"].get("pseudo_rescue_per_class", 16))
            _rescue_margin = float(cfg["training"].get("pseudo_rescue_margin", 0.05))
            _rescue_weight = float(cfg["training"].get("pseudo_rescue_weight", 0.1))
            _shape_np = _shape_probs.numpy() if use_shape_rescue else None
            _view_probs = np.stack([
                torch.softmax(_context_all, dim=1).numpy(),
                torch.softmax(_spectral_all, dim=1).numpy(),
                torch.softmax(_fused_all, dim=1).numpy(),
            ]) if use_shape_rescue else None
            _prototypes_t, _prototype_counts = _source_prototypes()
            _prototypes = _prototypes_t.cpu().numpy()
            _prototype_present = _prototype_counts.cpu().numpy() > 0
            _current_target_labels = _student_fused_pl.numpy()
            _selected_idx, _selected_y, _selected_w = [], [], []
            _main_counts = np.zeros(num_classes, dtype=np.int64)
            _rescue_counts = np.zeros(num_classes, dtype=np.int64)
            _main_index_set = set()
            for _c in range(num_classes):
                _c_idx = np.where((_pl == _c) & (_all_agree)
                                  & (_conf >= _thr) & (_weights > 0)
                                  & _teacher_ok
                                  & (_student_fused_pl == _c))[0]
                if len(_c_idx) > 0:
                    _sorted_idx = _c_idx[np.argsort(-(_weights[_c_idx] * _conf[_c_idx]))]
                    _sorted_idx = _sorted_idx[:_per_class]
                    _selected_idx.extend(_sorted_idx.tolist())
                    _selected_y.extend([_c] * len(_sorted_idx))
                    _selected_w.extend([1.0] * len(_sorted_idx))
                    _main_index_set.update(int(i) for i in _sorted_idx.tolist())
                    _main_counts[_c] = len(_sorted_idx)

                # Rescue only classes with weak main support. Require a confident,
                # margin-separated consensus probability and use a low loss weight.
                if _main_counts[_c] < _rescue_min_main and _rescue_per_class > 0:
                    _pc = _probs[:, _c].numpy()
                    _other = _probs.numpy().copy()
                    _other[:, _c] = -np.inf
                    _margin = _pc - np.max(_other, axis=1)
                    if use_shape_rescue:
                        _r_idx = select_anchored_rescue_indices(
                            _view_probs, _target_features_all, _prototypes,
                            _prototype_present, _c, _current_target_labels,
                            previous_target_labels if previous_target_labels is not None
                            else np.full(len(_current_target_labels), -1, dtype=np.int64),
                            min_view_confidence=float(cfg["training"].get(
                                "anchored_rescue_view_threshold", 0.60)),
                            min_prototype_similarity=float(cfg["training"].get(
                                "anchored_rescue_prototype_similarity", 0.50)),
                            limit=_rescue_per_class)
                        _score = (_target_features_all @ _prototypes[_c]) * _view_probs[:, :, _c].max(axis=0)
                    else:
                        _r_idx = np.where((_agreement) & (_weights > 0)
                                          & _teacher_ok
                                          & (_student_fused_pl == _c)
                                          & _teacher_rescue_ok
                                          & (_student_fused_conf >= _rescue_thr)
                                          & (_margin >= _rescue_margin))[0]
                        _score = _pc * _weights
                    _r_idx = np.asarray([i for i in _r_idx if int(i) not in _main_index_set])
                    if len(_r_idx) > 0:
                        _r_idx = _r_idx[np.argsort(-_score[_r_idx])]
                        _r_idx = _r_idx[:_rescue_per_class]
                        _selected_idx.extend(_r_idx.tolist())
                        _selected_y.extend([_c] * len(_r_idx))
                        _selected_w.extend([
                            float(cfg["training"].get("spectral_shape_rescue_weight", _rescue_weight))
                            if use_shape_rescue else _rescue_weight
                        ] * len(_r_idx))
                        _rescue_counts[_c] = len(_r_idx)

            if len(_selected_idx) > 0:
                _selected_idx = np.asarray(_selected_idx)
                pseudo_state["xb"] = tgt_transport_p[_selected_idx]
                pseudo_state["yb"] = np.asarray(_selected_y, dtype=np.int64)
                pseudo_state["weights"] = np.asarray(_selected_w, dtype=np.float32)
                pseudo_state["counts"] = np.bincount(pseudo_state["yb"], minlength=num_classes).tolist()
                pseudo_state["weight_sums"] = np.bincount(
                    pseudo_state["yb"], weights=pseudo_state["weights"], minlength=num_classes).tolist()
                pseudo_state["class_indices"] = {
                    int(c): np.where(pseudo_state["yb"] == c)[0]
                    for c in np.unique(pseudo_state["yb"])
                }
                _counts = np.bincount(pseudo_state["yb"], minlength=num_classes).tolist()
                _log(f"  [Pseudo] main={int(_main_counts.sum())} rescue={int(_rescue_counts.sum())} "
                     f"dist={_counts} main_dist={_main_counts.tolist()} "
                     f"rescue_dist={_rescue_counts.tolist()}")
            else:
                pseudo_state["xb"] = None
                pseudo_state["yb"] = None
                pseudo_state["weights"] = None
                pseudo_state["class_indices"] = None
                pseudo_state["counts"] = [0] * num_classes
                pseudo_state["weight_sums"] = [0.0] * num_classes
                _log(f"  [Pseudo] None selected (thr={_thr}, all-three agreement required)")
            previous_target_labels = _current_target_labels.copy()

        src_mmd_idx = _class_balanced_subsample(src_tr_y, 128,
                                                seed=cfg["output"]["seed"] + 500 + cycle)
        src_for_mmd = torch.from_numpy(src_tr_p[src_mmd_idx]).to(device)
        src_mmd_labels = torch.from_numpy(src_tr_y[src_mmd_idx]).long().to(device)
        # T7: 每 cycle 从 transport pool 随机抽 128，而非固定前 128
        tgt_desc = spectral_descriptor(
            torch.from_numpy(tgt_transport_p[_tgt_idx]).to(device))
        _mmd_pos = _tgt_idx if use_pseudo else np.arange(len(_tgt_idx))
        _target_probs = _probs[_mmd_pos].to(device)
        _target_weights_mmd = _target_weights[_mmd_pos].to(device)
        g_D, mmd0 = domain_gradient_mmd(
            transform, src_for_mmd, tgt_desc, source_labels=src_mmd_labels,
            target_class_probs=_target_probs, target_weights=_target_weights_mmd,
            min_effective_samples=int(cfg["tcst"].get("target_min_effective_samples", 4)))
        g_D_np = g_D.cpu().numpy().astype(np.float32)
        g_D_norm_raw = float(np.linalg.norm(g_D_np))
        _log(f"  [transport] MMD^2(phi=0)={mmd0:.4f}  ||g_D_raw||={g_D_norm_raw:.4e} "
             f"(raw, per doc 5.3)")

        sel_idx, sel_y, sel_k1, sel_k2 = select_constraint_samples(
            model, src_cn_p, src_cn_y, device,
            per_class=cfg["tcst"]["per_class_constraint_samples"],
            top_competitors=cfg["tcst"].get("top_competitors", 2))

        if len(sel_idx) == 0:
            phi_star = torch.zeros(transform.dim, device=device)
            diag = {"skipped": "no_constraint_samples"}
            _log("  [transport] skipped (no constraint samples)")
        else:
            G_np, m0_np, w_np_unused, j_class_np, comp_class_np = compute_margin_gradients(
                model, transform, src_cn_p[sel_idx.numpy()], sel_y, sel_k1, sel_k2, device)
            
            if use_rs_crw and cycle > 1:
                w_np = class_risk_weights[j_class_np].astype(np.float32)
                _log(f"  [RS-CRW] w_j (min={w_np.min():.2f}, max={w_np.max():.2f})")
            else:
                w_np = np.ones(len(m0_np), dtype=np.float32)

            _log(f"  [transport] margin J={G_np.shape[0]}  mean_m0={m0_np.mean():.3f}")
            _use_qp = bool(cfg["tcst"].get("use_qp", True))
            if _use_qp:
                qp = solve_margin_qp(g_D_np, G_np, m0_np, w_np,
                                     rho_m=cfg["tcst"]["rho_m"],
                                     lambda_d=cfg["tcst"]["lambda_d"],
                                     lambda_xi=cfg["tcst"]["lambda_xi"])
            else:
                # 消融 I1: 直接用 -g_D / ||g_D||
                _g_norm = float(np.linalg.norm(g_D_np))
                _d_raw = (-g_D_np / max(_g_norm, 1e-8)).astype(np.float32)
                qp = {"status": "raw_gradient", "d": _d_raw,
                      "directional_derivative": float(g_D_np @ _d_raw),
                      "d_norm": float(np.linalg.norm(_d_raw)),
                      "slack_mean": 0.0, "slack_max": 0.0,
                      "num_constraints": int(G_np.shape[0])}
            _log(f"  [qp] {qp['status']}  ||d*||={qp['d_norm']:.4f}  "
                 f"<g_D,d*>={qp['directional_derivative']:+.4f}  "
                 f"slack_max={qp['slack_max']:.4f}")
            # I1 fix: raw_gradient 也允许进入 Risk Search（不是拦截）
            _acceptable_status = ("optimal", "optimal_inaccurate", "no_constraints", "raw_gradient")
            if qp["status"] not in _acceptable_status or qp["d_norm"] < 1e-8:
                phi_star = torch.zeros(transform.dim, device=device)
                diag = {"qp_status": qp["status"], "d_norm": qp["d_norm"]}
            else:
                _use_risk = bool(cfg["tcst"].get("use_risk_search", True))
                if not _use_risk:
                    # 消融 I2: 不做回溯，直接接受 eta=1.0
                    eta_star = 1.0
                    phi_star = torch.from_numpy(qp["d"]).to(device)
                    diag = {"accepted_eta": 1.0, "skipped_risk_search": True,
                            "candidates": [], "accepted_candidate": None}
                    _log(f"  [risk] DISABLED: direct accept eta=1.0  "
                         f"phi_norm={float(phi_star.norm()):.4f}")
                else:
                    eta_star, phi_star, diag = backtracking_search(
                        model, transform, qp["d"],
                        src_ca_p, src_ca_y,
                        tgt_calibration_p,
                        eta_candidates=cfg["tcst"]["eta_candidates"],
                    tau_sem=cfg["tcst"]["tau_sem"],
                    tau_flip=cfg["tcst"]["tau_flip"],
                    tau_class_flip=cfg["tcst"]["tau_class_flip"],
                    tau_swd_rel=cfg["tcst"].get("tau_swd_rel", 0.01),
                    device=device,
                    swd_seed=cfg["output"]["seed"] + cycle,
                    gain_tv_max=float(cfg["tcst"].get("gain_tv_max", 0.36)),
                    offset_tv_max=float(cfg["tcst"].get("offset_tv_max", 0.18)),
                    rel_change_max=float(cfg["tcst"].get("rel_change_max", 0.15)),
                    rsem_delta=float(cfg["tcst"].get("rsem_delta", 0.05)),
                    domain_cal_patches=src_domain_cal_p)
                _log(f"  [risk] accepted eta={eta_star:.4f}  "
                     f"phi_norm={float(phi_star.norm()):.4f}")
                diag["accepted_eta"] = float(eta_star)

        rng2 = np.random.default_rng(cfg["output"]["seed"] + 100 + cycle)
        if cb_power > 0:
            cycle_loader = _make_balanced_loader(src_tr_p, src_tr_y, class_weights_for_ce,
                                                 batch, seed=cfg["output"]["seed"] + 100 + cycle)
            def _iter_cycle():
                for xb_batch, yb_batch in cycle_loader:
                    yield xb_batch, yb_batch
        else:
            def _iter_cycle():
                idx = rng2.permutation(len(src_tr_p))
                for s in range(0, len(idx), batch):
                    sel = idx[s:s+batch]
                    yield src_tr_p[sel], src_tr_y[sel]

        for ep in range(1, cfg["training"]["epochs_per_cycle"] + 1):
            losses = []
            for xb_batch, yb_batch in _iter_cycle():
                losses.append(_train_step(xb_batch, yb_batch, phi_star,
                                          cfg["tcst"]["lambda_safe"]))
            # P1.3: 统计本 epoch 的 safe 执行次数
            _safe_count = sum(1 for l in losses if l.get("safe_executed", False))
            _safe_losses = [l["safe_loss"] for l in losses if l.get("safe_loss") is not None]
            _clean_losses = [l["clean_loss"] for l in losses]
            _pseudo_draw_counts = np.sum([
                l["pseudo_draw_counts"] for l in losses
                if l["pseudo_draw_counts"] is not None
            ], axis=0).astype(int).tolist() if any(
                l["pseudo_draw_counts"] is not None for l in losses) else [0] * num_classes
            _pseudo_repeats = sum(l["pseudo_repeated_draws"] for l in losses)
            src_val, _cyc_details = _eval_source(
                model, src_val_p, src_val_y, device,
                use_cb_val=cfg["training"].get("use_cb_val", False),
                ckpt_mode=ckpt_mode, num_classes=num_classes,
                target_patches=tgt_transport_p[:256] if ckpt_mode == "B" else None,
                return_details=True)
            src_nll = _cyc_details["nll"]
            # P1.1: 记录每个 cycle epoch 的元数据
            _aug_count_cyc = sum(1 for l in losses if l.get("aug_executed", False))
            _aug_losses = [l["aug_loss"] for l in losses if l.get("aug_loss") is not None]
            _cyc_entry = {
                "cycle": cycle, "epoch": ep, "src_val": float(src_val), "src_nll": float(src_nll),
                "loss_mean": float(np.mean([l["loss"] for l in losses])),
                "clean_loss_mean": float(np.mean(_clean_losses)),
                "safe_loss_mean": float(np.mean(_safe_losses)) if _safe_losses else None,
                "aug_loss_mean": float(np.mean(_aug_losses)) if _aug_losses else None,
                "safe_executed_count": _safe_count,
                "aug_executed_count": _aug_count_cyc,
                "safe_total_batches": len(losses),
                "pseudo_draw_counts": _pseudo_draw_counts,
                "pseudo_repeated_draws": int(_pseudo_repeats),
                "is_best": False,
            }
            if src_val > best_global["src_val"]:
                best_global["src_val"] = src_val
                best_global["state"] = copy.deepcopy(model.state_dict())
                best_global["source"] = "cycle"
                best_global["cycle"] = cycle
                best_global["epoch"] = ep
                _cyc_entry["is_best"] = True
                marker = " *BEST*"
            else:
                marker = ""
            cycle_ckpt_manifest.append(_cyc_entry)

            # P1.2: D 模式选择（合格条件 + NLL 排序，与 best_global 独立）
            if ckpt_mode == "D":
                if src_val >= _d_base_src_val - _d_tolerance:
                    if src_nll < _d_best_nll:
                        _d_best_nll = src_nll
                        _d_best_state = copy.deepcopy(model.state_dict())
                        _d_best_meta = {"cycle": cycle, "epoch": ep,
                                        "src_val": float(src_val), "src_nll": float(src_nll)}
            _log(f"    cyc{cycle} ep{ep:02d}  loss={np.mean([l['loss'] for l in losses]):.4f}  "
                 f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                 f"src_val={src_val:.2f}  best={best_global['src_val']:.2f}  "
                 f"aug={_aug_count_cyc}/{len(losses)}  safe={_safe_count}/{len(losses)}  "
                 f"pseudo_draws={_pseudo_draw_counts} repeats={_pseudo_repeats}{marker}")
            if scheduler is not None:
                scheduler.step(src_val) if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else scheduler.step()

        if use_teacher_consistency:
            _update_teacher()

        if use_rs_crw:
            cands = diag.get("candidates", [])
            cal_counts = diag.get("cal_class_counts", None)
            accepted_cand = diag.get("accepted_candidate", None)
            if (accepted_cand is not None and cal_counts is not None
                    and float(accepted_cand.get("eta", 0.0)) > 1e-8):
                Vc = accepted_cand.get("Vc", None)
                if Vc is not None:
                    new_w = class_risk_weights.copy()
                    upd = []
                    for c in range(num_classes):
                        if cal_counts[c] >= rs_crw_min_cal and c < len(Vc):
                            v = float(Vc[c])
                            target_w = 1.0 + rs_crw_alpha * v
                            new_w[c] = (1.0 - rs_crw_ema) * class_risk_weights[c] + rs_crw_ema * target_w
                            upd.append(c)
                    class_risk_weights = new_w
                    _log(f"  [RS-CRW] updated weights (classes={upd}): "
                         f"{[round(float(x), 2) for x in class_risk_weights.tolist()]}")
                else:
                    _log(f"  [RS-CRW] skip (no Vc)")
            else:
                _log(f"  [RS-CRW] skip (eta=0 or no candidate)")

        # A12 fix: persist phi_star / d_star stats for post-hoc audit
        # B3 fix: 保存完整 phi_star / d_star 向量（不只 stats）
        _a12_phi_stats = None
        _a12_d_stats = None
        _a12_phi_full = None
        _a12_d_full = None
        try:
            if 'phi_star' in dir() and phi_star is not None:
                _pf = phi_star.detach().cpu().numpy().flatten()
                _a12_phi_stats = {
                    "norm": float(np.linalg.norm(_pf)),
                    "abs_max": float(np.abs(_pf).max()) if _pf.size else 0.0,
                    "mean": float(_pf.mean()) if _pf.size else 0.0,
                    "std": float(_pf.std()) if _pf.size else 0.0,
                }
                _a12_phi_full = [float(x) for x in _pf.tolist()]
        except Exception:
            pass
        try:
            if 'qp' in dir() and isinstance(qp, dict) and qp.get('d') is not None:
                _d = np.asarray(qp['d']).flatten()
                _a12_d_stats = {
                    "norm": float(np.linalg.norm(_d)),
                    "abs_max": float(np.abs(_d).max()) if _d.size else 0.0,
                    "status": qp.get('status'),
                }
                _a12_d_full = [float(x) for x in _d.tolist()]
        except Exception:
            pass

        # P1.3: 本 cycle 的 safe 执行统计
        _cycle_safe_counts = [e for e in cycle_ckpt_manifest if e["cycle"] == cycle]
        _total_safe = sum(e["safe_executed_count"] for e in _cycle_safe_counts)
        _total_batches = sum(e["safe_total_batches"] for e in _cycle_safe_counts)
        # P2: 记录目标域无标签预测分布（只用无标签数据，不碰 target GT）
        _tgt_pred_dist = None
        try:
            _pred_model = model
            _pred_model.eval()
            _tgt_probs_list = []
            with torch.no_grad():
                _n_eval = min(512, len(tgt_transport_p))
                _tgt_eval_idx = np.random.default_rng(cfg["output"]["seed"] + 999 + cycle).choice(
                    len(tgt_transport_p), _n_eval, replace=False)
                for _s in range(0, _n_eval, 128):
                    _batch_idx = _tgt_eval_idx[_s:_s+128]
                    _xb = torch.from_numpy(tgt_transport_p[_batch_idx]).to(device)
                    _lg = _pred_model(_xb).logits.float().cpu()
                    _tgt_probs_list.append(torch.softmax(_lg, dim=1))
                _tgt_probs = torch.cat(_tgt_probs_list, 0)
                _pred_counts = _tgt_probs.argmax(1).numpy()
                _dist = np.bincount(_pred_counts, minlength=num_classes).astype(np.float64)
                _dist = _dist / max(1.0, _dist.sum())
                # 预测熵（无标签辅助指标）
                _ent = -torch.sum(_tgt_probs * torch.log(_tgt_probs + 1e-8), dim=1).mean().item()
                _tgt_pred_dist = {
                    "pred_class_dist": _dist.tolist(),
                    "pred_entropy": float(_ent),
                    "n_samples": int(_n_eval),
                }
        except Exception as e:
            _log(f"  [P2] target pred dist failed: {e}")

        cycle_logs.append({"cycle": cycle, "diag": diag,
                            "best_src_val": float(best_global["src_val"]),
                            "best_source": best_global.get("source", "warmup"),
                            "phi_star_stats": _a12_phi_stats,
                            "d_star_stats": _a12_d_stats,
                            "phi_star_full": _a12_phi_full,
                            "d_star_full": _a12_d_full,
                            "accepted_eta": float(diag.get("accepted_eta", 0.0)),
                            "safe_executed_total": int(_total_safe),
                            "safe_batches_total": int(_total_batches),
                            "source_rollback": False,
                            "tgt_pred_dist": _tgt_pred_dist,
                            "class_risk_weights": [float(x) for x in class_risk_weights.tolist()] if use_rs_crw else None})
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"stage": "cycle", "cycle": cycle,
                                "transport": diag,
                                "pseudo_pool_counts": pseudo_state.get("counts", [0] * num_classes),
                                "pseudo_pool_weight_sums": pseudo_state.get("weight_sums", [0.0] * num_classes),
                                "phi_star_stats": _a12_phi_stats,
                                "d_star_stats": _a12_d_stats,
                                "phi_star_full": _a12_phi_full,
                                "d_star_full": _a12_d_full,
                                "tgt_pred_dist": _tgt_pred_dist,
                                "class_risk_weights": [float(x) for x in class_risk_weights.tolist()] if use_rs_crw else None
                                }, default=str) + "\n")

    force_final = bool(cfg["training"].get("force_final_epoch", False))
    _d_selected = False
    if ckpt_mode == "D" and _d_best_state is not None:
        model.load_state_dict(_d_best_state)
        best_global["src_val"] = _d_best_meta["src_val"]
        best_global["source"] = "D_cycle"
        best_global["cycle"] = _d_best_meta["cycle"]
        best_global["epoch"] = _d_best_meta["epoch"]
        _d_selected = True
        _log(f"\n[P1.2] ckpt_mode=D: selected cycle {_d_best_meta['cycle']} ep {_d_best_meta['epoch']} "
             f"(src_val={_d_best_meta['src_val']:.2f}, nll={_d_best_meta['src_nll']:.4f}, "
             f"tol={_d_tolerance}, base={_d_base_src_val:.2f})")
    elif ckpt_mode == "D" and _d_best_state is None:
        _log(f"\n[P1.2] ckpt_mode=D: no qualified cycle checkpoint "
             f"(all cycles have src_val < {_d_base_src_val - _d_tolerance:.2f}), keeping warmup")

    if not _d_selected:
        if force_final:
            _log(f"\n[Opt3] force_final_epoch=True, keeping final model "
                 f"(src_val={best_global['src_val']:.2f} at best)")
        else:
            model.load_state_dict(best_global["state"])
            _log(f"\n[Phase C] restored best src_val checkpoint "
                 f"(src_val={best_global['src_val']:.2f})")

    # P1.1 + P1.3: 在 run_tcst_v8 里构造完整 manifest，传给 _final_eval
    _manifest_data = {
        "warmup_manifest": warmup_ckpt_manifest,
        "cycle_manifest": cycle_ckpt_manifest,
        "final_selected_source": best_global.get("source", "warmup"),
        "final_selected_cycle": best_global.get("cycle", None),
        "final_selected_epoch": best_global.get("epoch", None),
        "final_selected_src_val": float(best_global.get("src_val", -1)),
        "warmup_best_src_val": float(best_src_val),
        "force_final_epoch": bool(cfg["training"].get("force_final_epoch", False)),
        "ckpt_mode": cfg["training"].get("ckpt_mode", "A"),
        "ckpt_d_tolerance": _d_tolerance if ckpt_mode == "D" else None,
        "ckpt_d_base_src_val": _d_base_src_val if ckpt_mode == "D" else None,
        "ckpt_d_selected": {
            "cycle": _d_best_meta["cycle"], "epoch": _d_best_meta["epoch"],
            "src_val": _d_best_meta["src_val"], "src_nll": _d_best_meta["src_nll"],
        } if _d_best_meta is not None else None,
    }
    return _final_eval(model, transform, cfg, device, save_dir,
                       best_global["src_val"], stats, cycle_logs, ds_name,
                       num_classes, bands, tgt_cube, patch_size,
                       manifest_data=_manifest_data,
                       best_state_for_diff=best_state)


def _final_eval(model, transform, cfg, device, save_dir, best_src_val,
                stats, cycle_logs, ds_name, num_classes, bands, tgt_cube, patch_size,
                manifest_data=None, best_state_for_diff=None):
    """Load target GT NOW (after training) and evaluate."""
    _log(f"\n[Final] loading target GT for evaluation ONLY (principle 4)")
    tgt_gt = _load_gt_map(cfg["dataset"]["target_gt"])
    tgt_valid = np.linalg.norm(tgt_cube, axis=-1) > 0
    tgt_eval_c = np.argwhere((tgt_gt > 0) & tgt_valid)
    tgt_eval_y = (tgt_gt[tgt_eval_c[:, 0], tgt_eval_c[:, 1]] - 1).astype(np.int64)
    tgt_eval_p = build_patches(tgt_cube, tgt_eval_c, patch_size)
    _log(f"[target-eval] usable GT pixels = {len(tgt_eval_c)}")

    metrics = _eval_target(model, tgt_eval_p, tgt_eval_y, device, num_classes, use_tta=True)
    result = {"dataset": ds_name, "bands": int(bands),
              "num_classes": int(num_classes), "cycles": cycle_logs,
              "best_src_val": float(best_src_val),
              "source_split": stats,
              "target_metrics": metrics}

    _log(f"\n{'='*70}\nTARGET RESULT — {ds_name}\n{'='*70}")
    _log(f"OA     = {metrics['OA']:.2f}")
    _log(f"AA     = {metrics['AA']:.2f}")
    _log(f"Kappa  = {metrics['Kappa']:.4f}")
    _log(f"MacroF1= {metrics['MacroF1']:.2f}")
    _print_per_class(metrics, num_classes, ds_name)
    _log(f"{'='*70}")

    # P1.1 + P1.3: manifest 由 run_tcst_v8 传入，不再从 dir() 读局部变量
    _manifest = manifest_data if manifest_data is not None else {}
    (save_dir / "checkpoint_manifest.json").write_text(
        json.dumps(_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    torch.save({"model": model.state_dict(), "transform": transform.state_dict(),
                "args": cfg, "metrics": metrics},
               save_dir / "final_state.pt")
    (save_dir / "final_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    # P1.3: 计算 final vs warmup 的参数差异范数（用传入的 best_state_for_diff）
    _param_diff = None
    if best_state_for_diff is not None:
        try:
            import torch as _torch
            _final_sd = model.state_dict()
            _diffs = []
            for k in best_state_for_diff.keys():
                if k in _final_sd and _final_sd[k].dtype.is_floating_point:
                    _diffs.append(_torch.norm(_final_sd[k].float() - best_state_for_diff[k].float()).item())
            _param_diff = float(sum(_diffs))
        except Exception as e:
            _log(f"[P1.3] param diff calculation failed: {e}")

    if _param_diff is not None:
        _log(f"[P1.3] param diff (warmup best -> final): L1-sum-of-L2 = {_param_diff:.4f}")
        _manifest["param_diff_warmup_to_final"] = _param_diff
        (save_dir / "checkpoint_manifest.json").write_text(
            json.dumps(_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    _log(f"Saved: {save_dir / 'final_metrics.json'}")
    return result
