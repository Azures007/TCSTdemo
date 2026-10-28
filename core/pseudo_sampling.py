"""Small sampling helpers for target pseudo-label batches."""
from __future__ import annotations

import numpy as np


def sample_pseudo_indices(labels, batch_size, rng):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    batch_size = int(batch_size)
    if labels.size == 0 or batch_size <= 0:
        raise ValueError("labels and batch_size must be non-empty and positive")
    classes, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    class_draws = rng.choice(len(classes), size=batch_size, p=np.sqrt(counts) / np.sqrt(counts).sum())
    indices = np.empty(batch_size, dtype=np.int64)
    for class_id, label in enumerate(classes):
        positions = np.flatnonzero(inverse == class_id)
        selected = np.flatnonzero(class_draws == class_id)
        if selected.size:
            indices[selected] = rng.choice(positions, selected.size, replace=True)
    return indices


def select_shape_rescue_indices(shape_probs, view_probs, class_id,
                                min_shape_confidence, min_shape_margin,
                                min_view_confidence, limit):
    shape_probs = np.asarray(shape_probs, dtype=np.float64)
    view_probs = np.asarray(view_probs, dtype=np.float64)
    class_id, limit = int(class_id), int(limit)
    if (shape_probs.ndim != 2 or view_probs.ndim != 3
            or view_probs.shape[1:] != shape_probs.shape
            or not 0 <= class_id < shape_probs.shape[1]
            or limit < 0 or not np.isfinite(shape_probs).all()
            or not np.isfinite(view_probs).all()):
        raise ValueError("invalid shape/view probabilities or class/limit")
    other_shape = np.delete(shape_probs, class_id, axis=1).max(axis=1)
    shape_margin = shape_probs[:, class_id] - other_shape
    view_support = ((view_probs[:, :, class_id] >= min_view_confidence)
                    & (view_probs.argmax(axis=2) == class_id)).any(axis=0)
    candidates = np.flatnonzero(
        (shape_probs.argmax(axis=1) == class_id)
        & (shape_probs[:, class_id] >= min_shape_confidence)
        & (shape_margin >= min_shape_margin) & view_support)
    score = shape_probs[candidates, class_id] * view_probs[:, candidates, class_id].max(axis=0)
    return candidates[np.argsort(-score)[:limit]]
