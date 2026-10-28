"""Small sampling helpers for target pseudo-label batches."""
from __future__ import annotations

import numpy as np


def sample_pseudo_indices(labels, batch_size, rng, mode="class_balanced"):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    batch_size = int(batch_size)
    if labels.size == 0 or batch_size <= 0:
        raise ValueError("labels and batch_size must be non-empty and positive")
    if mode == "pool_uniform":
        return rng.choice(labels.size, batch_size, replace=True)
    if mode != "class_balanced":
        raise ValueError(f"unknown pseudo sampling mode: {mode}")
    classes, inverse = np.unique(labels, return_inverse=True)
    class_draws = rng.integers(len(classes), size=batch_size)
    indices = np.empty(batch_size, dtype=np.int64)
    for class_id, label in enumerate(classes):
        positions = np.flatnonzero(inverse == class_id)
        selected = np.flatnonzero(class_draws == class_id)
        if selected.size:
            indices[selected] = rng.choice(positions, selected.size, replace=True)
    return indices
