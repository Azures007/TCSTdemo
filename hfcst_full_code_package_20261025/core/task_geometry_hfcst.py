"""Margin-geometry sample selection and gradients for HFCST transport parameters."""
from __future__ import annotations
import numpy as np
import torch


def _logits(model, patches, device, batch_size=128):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            x = torch.as_tensor(patches[i:i+batch_size], dtype=torch.float32, device=device)
            outs.append(model(x).logits.float().cpu())
    return torch.cat(outs, dim=0) if outs else torch.empty((0, getattr(model, "num_classes", 0)))


def select_constraint_samples(model, patches, labels, device, per_class=12, stable_fraction=0.25):
    """Select correctly classified boundary and stable samples, class by class."""
    y = np.asarray(labels, dtype=np.int64)
    if len(y) == 0 or len(patches) == 0:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, empty
    logits = _logits(model, patches, device).numpy()
    pred = logits.argmax(axis=1)
    true = logits[np.arange(len(y)), y]
    masked = logits.copy()
    masked[np.arange(len(y)), y] = -np.inf
    competitor = masked.argmax(axis=1)
    margin = true - masked.max(axis=1)
    chosen = []
    for c in np.unique(y):
        idx = np.flatnonzero((y == c) & (pred == y))
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(margin[idx])]
        take = min(int(per_class), len(idx))
        if take == len(idx):
            sel = idx
        else:
            n_boundary = max(1, int(round(take * (1.0 - stable_fraction))))
            n_stable = max(0, take - n_boundary)
            left = idx[:n_boundary]
            right = idx[-n_stable:] if n_stable else np.empty(0, dtype=int)
            sel = np.unique(np.concatenate([left, right]))
            if len(sel) < take:
                remaining = np.setdiff1d(idx, sel, assume_unique=False)
                sel = np.concatenate([sel, remaining[:take-len(sel)]])
        chosen.extend(sel.tolist())
    chosen = np.asarray(sorted(set(chosen)), dtype=np.int64)
    return (torch.from_numpy(chosen), torch.from_numpy(y[chosen]),
            torch.from_numpy(competitor[chosen].astype(np.int64)))


def compute_margin_gradients(model, transform, patches, labels, competitors, device,
                             parameter_kind="shared"):
    """Return G and current margins for hard linearized margin constraints.

    parameter_kind='shared' returns [N, 2K]. 'residual' returns [N, C*2K].
    Non-finite rows are removed; the fourth return value maps kept rows back to inputs.
    """
    y = torch.as_tensor(labels, dtype=torch.long, device=device).reshape(-1)
    k = torch.as_tensor(competitors, dtype=torch.long, device=device).reshape(-1)
    if len(patches) != len(y) or len(y) != len(k):
        raise ValueError("patches, labels, competitors must have equal lengths")
    if parameter_kind == "shared":
        param = transform.phi_shared
    elif parameter_kind == "residual":
        param = transform.delta_phi
    else:
        raise ValueError("parameter_kind must be 'shared' or 'residual'")
    rows, margins = [], []
    model.eval()
    for i in range(len(y)):
        xb = torch.as_tensor(patches[i:i+1], dtype=torch.float32, device=device)
        yi, ki = y[i:i+1], k[i:i+1]
        transformed = transform(xb, labels=yi)
        logits = model(transformed).logits.float()
        margin = logits[0, yi[0]] - logits[0, ki[0]]
        grad = torch.autograd.grad(margin, param, retain_graph=False, create_graph=False,
                                   allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(param)
        rows.append(grad.reshape(-1).detach().cpu().numpy())
        margins.append(float(margin.detach().cpu().item()))
    dim = transform.dim if parameter_kind == "shared" else transform.num_classes * transform.dim
    G = np.asarray(rows, dtype=np.float64).reshape(-1, dim)
    m = np.asarray(margins, dtype=np.float64)
    finite = np.isfinite(G).all(axis=1) & np.isfinite(m)
    valid_rows = np.flatnonzero(finite)
    return G[finite], m[finite], int((~finite).sum()), valid_rows
