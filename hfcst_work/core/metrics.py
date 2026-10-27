"""Classification metrics."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes):
    m = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            m[t, p] += 1
    return m


def classification_metrics(m):
    total = int(m.sum())
    correct = int(np.trace(m))
    oa = 100.0 * correct / max(1, total)
    row = m.sum(1)
    class_acc = np.divide(np.diag(m), row, out=np.zeros_like(row, dtype=np.float64),
                          where=row > 0)
    aa = 100.0 * float(class_acc.mean())
    col = m.sum(0)
    expected = float(np.dot(row, col)) / max(1.0, float(total * total))
    obs = float(correct) / max(1.0, float(total))
    kappa = (obs - expected) / max(1e-12, 1.0 - expected)

    # macro F1
    f1s = []
    for c in range(m.shape[0]):
        tp = m[c, c]
        fp = m[:, c].sum() - tp
        fn = m[c, :].sum() - tp
        if tp + fp + fn == 0:
            continue
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    macro_f1 = 100.0 * float(np.mean(f1s)) if f1s else 0.0

    return {"OA": oa, "AA": aa, "Kappa": float(kappa), "MacroF1": macro_f1,
            "class_accuracy": (100.0 * class_acc).tolist(),
            "confusion_matrix": m.tolist()}
