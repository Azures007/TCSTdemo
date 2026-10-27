"""Source Constraint Set + Top-2 Pairwise Margin + phi-gradients (v8.0.1).

v8.0.1 修复: 每个约束样本对两个最强竞争类别分别建立 Margin 约束。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def select_constraint_samples(model, patches, labels, device,
                              per_class=12, top_competitors=2):
    """Return (idx, y_true, k1, k2) for correctly-classified source patches,
    class-balanced, with top-2 competing classes per sample.
    """
    model.eval()
    idx = np.arange(len(patches))
    sel_idx, sel_y, sel_k1, sel_k2 = [], [], [], []
    with torch.no_grad():
        logits_all = []
        for start in range(0, len(patches), 256):
            batch = torch.from_numpy(patches[start:start+256]).to(device)
            logits_all.append(model(batch).logits.float())
        logits = torch.cat(logits_all, 0).cpu()
    pred = logits.argmax(1)
    for c in range(int(labels.max()) + 1):
        mask = (labels == c) & (pred.numpy() == c)
        ids = idx[mask]
        if len(ids) == 0:
            continue
        margins = []
        for i in ids:
            z = logits[i]
            true_logit = z[c].item()
            other = z.clone(); other[c] = -1e9
            topk = other.topk(top_competitors)
            margins.append(true_logit - topk.values[0].item())
        margins = np.asarray(margins)
        order = np.argsort(margins)
        n = min(per_class, len(ids))
        # B2 fix: n=1 时 half=0 会选 0 个样本；改为 (n+1)//2 保底 1 个
        n_boundary = (n + 1) // 2
        n_easy = n - n_boundary
        chosen_list = list(order[:n_boundary])
        if n_easy > 0:
            chosen_list.extend(order[-n_easy:].tolist())
        chosen = np.asarray(chosen_list, dtype=int)
        assert len(chosen) == n, f"约束选择数量错误: {len(chosen)} != {n}"
        assert len(np.unique(chosen)) == n, f"约束样本有重复"
        for i_local in chosen:
            gi = ids[int(i_local)]
            z = logits[gi]
            other = z.clone(); other[c] = -1e9
            topk = other.topk(top_competitors)
            k1 = int(topk.indices[0])
            k2 = int(topk.indices[1]) if top_competitors >= 2 and len(topk.indices) >= 2 else k1
            sel_idx.append(int(gi))
            sel_y.append(int(c))
            sel_k1.append(k1)
            sel_k2.append(k2)
    return (torch.as_tensor(sel_idx, dtype=torch.long),
            torch.as_tensor(sel_y, dtype=torch.long),
            torch.as_tensor(sel_k1, dtype=torch.long),
            torch.as_tensor(sel_k2, dtype=torch.long))


def compute_margin_gradients(model, transform, patches, y_true, y_comp1, y_comp2, device,
                             batch_size=8):
    """For each sample compute grad_phi for BOTH top-1 and top-2 competitors.

    Returns:
        G [2J, 2K]  — two constraints per sample (k1, k2)
        m0 [2J]     — initial margins
        w  [2J]     — weights (all 1.0; RS-CRW reweights later)
        j_class [2J] — class id per constraint (for RS-CRW)
        comp_class [2J] — competitor id per constraint
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dim = transform.dim
    G_list, m0_list, w_list, jc_list, cc_list = [], [], [], [], []
    for start in range(0, len(patches), batch_size):
        end = min(start + batch_size, len(patches))
        for i in range(start, end):
            x = torch.from_numpy(patches[i:i+1]).to(device)
            y = int(y_true[i].item())
            for k in (int(y_comp1[i].item()), int(y_comp2[i].item())):
                phi = torch.zeros(dim, device=device, requires_grad=True)
                labels_t = torch.tensor([y], device=device, dtype=torch.long)
                x_phi = transform(x, phi, labels=labels_t)
                out = model(x_phi)
                logit = out.logits
                m = logit[0, y] - logit[0, k]
                grad = torch.autograd.grad(m, phi, create_graph=False)[0]
                G_list.append(grad.detach().cpu().numpy())
                m0_list.append(float(m.detach()))
                w_list.append(1.0)
                jc_list.append(y)
                cc_list.append(k)
    for p in model.parameters():
        p.requires_grad_(True)
    assert len(G_list) == len(m0_list) == len(w_list) == len(jc_list) == len(cc_list), \
        f"compute_margin_gradients 长度不一致: G={len(G_list)}, m0={len(m0_list)}, " \
        f"w={len(w_list)}, jc={len(jc_list)}, cc={len(cc_list)}"
    return (np.stack(G_list, 0).astype(np.float32),
            np.asarray(m0_list, dtype=np.float32),
            np.asarray(w_list, dtype=np.float32),
            np.asarray(jc_list, dtype=np.int64),
            np.asarray(cc_list, dtype=np.int64))
