"""P2 第 7 条：合成数据 MMD 误差测试（v2，修正结论）。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))

import numpy as np
import torch
from core.discrepancy import mk_mmd2


def make_class_data(means, covs, class_props, n_total, rng):
    n_classes = len(means)
    counts = np.round(np.array(class_props) * n_total).astype(int)
    X, Y = [], []
    for c in range(n_classes):
        n_c = counts[c]
        if n_c == 0: continue
        x = rng.multivariate_normal(means[c], covs[c], size=n_c)
        X.append(x); Y.append(np.full(n_c, c))
    X = np.concatenate(X); Y = np.concatenate(Y)
    perm = rng.permutation(len(X))
    return torch.from_numpy(X[perm].astype(np.float32)), Y[perm]


def class_conditional_mmd(X_src, Y_src, X_tgt, Y_tgt, n_classes=3):
    total = 0.0; used = 0
    for c in range(n_classes):
        s_c = X_src[Y_src == c]; t_c = X_tgt[Y_tgt == c]
        if len(s_c) < 5 or len(t_c) < 5: continue
        total += float(mk_mmd2(s_c, t_c)); used += 1
    return total / used if used else 0.0


def main():
    rng = np.random.default_rng(42)
    d = 50
    means = [np.zeros(d), np.ones(d) * 2, np.ones(d) * 4]
    covs = [np.eye(d) * 0.5] * 3

    print("=" * 90)
    print("  合成数据 MMD 误差测试（v2）")
    print("=" * 90)

    scenarios = []

    # 场景 1: 条件相同，比例不同
    src_1, y_s1 = make_class_data(means, covs, [0.5, 0.3, 0.2], 1000, rng)
    tgt_1, y_t1 = make_class_data(means, covs, [0.2, 0.3, 0.5], 1000, rng)
    mmd_1 = float(mk_mmd2(src_1, tgt_1))
    cc_1 = class_conditional_mmd(src_1, y_s1, tgt_1, y_t1)
    scenarios.append(("S1: 条件相同 / 比例不同", mmd_1, cc_1, "真域差异应为 0"))

    # 场景 2: 比例相同，条件偏移
    tgt_means_shifted = [m + 0.5 for m in means]
    src_2, y_s2 = make_class_data(means, covs, [0.33, 0.33, 0.34], 1000, rng)
    tgt_2, y_t2 = make_class_data(tgt_means_shifted, covs, [0.33, 0.33, 0.34], 1000, rng)
    mmd_2 = float(mk_mmd2(src_2, tgt_2))
    cc_2 = class_conditional_mmd(src_2, y_s2, tgt_2, y_t2)
    scenarios.append(("S2: 比例相同 / 条件偏移", mmd_2, cc_2, "条件偏移被边缘 MMD 低估"))

    # 场景 3: 比例+条件都变
    tgt_means_offset = [means[0] - 0.3, means[1] + 0.0, means[2] + 0.3]
    src_3, y_s3 = make_class_data(means, covs, [0.6, 0.2, 0.2], 1000, rng)
    tgt_3, y_t3 = make_class_data(tgt_means_offset, covs, [0.2, 0.2, 0.6], 1000, rng)
    mmd_3 = float(mk_mmd2(src_3, tgt_3))
    cc_3 = class_conditional_mmd(src_3, y_s3, tgt_3, y_t3)
    scenarios.append(("S3: 比例+条件都变", mmd_3, cc_3, "边缘 MMD 被比例差异主导"))

    print(f"\n  {'scenario':<32} {'edge MMD':>12} {'class-cond MMD':>16} {'edge/cc':>10}")
    print('-' * 80)
    for name, mmd, cc, note in scenarios:
        ratio = mmd / max(cc, 1e-8)
        print(f"  {name:<30} {mmd:>12.6f} {cc:>16.6f} {ratio:>10.2f}")
        print(f"    → {note}")

    print("\n" + "=" * 90)
    print("  结论（修正版）")
    print("=" * 90)
    print("  1. 边缘 MMD 无法区分'比例差异'和'条件偏移'，两个方向都会出错：")
    print(f"     - S1: 条件相同，边缘 MMD={mmd_1:.4f} ≠ 0（被比例差异污染）")
    print(f"     - S2: 条件偏移，边缘 MMD={mmd_2:.4f} 低估了类条件 MMD={cc_2:.4f}（{cc_2/max(mmd_2,1e-8):.1f}倍）")
    print(f"     - S3: 两者都有，边缘 MMD={mmd_3:.4f} 被比例差异主导，类条件 MMD={cc_3:.4f}")
    print("  2. 这直接解释 Houston 的现象：")
    print("     - Houston target 的 class5 占比 0.37-0.79，source 只 0.2")
    print("     - 边缘 MMD 降低可能来自 class5 的对齐（大类占比上升）")
    print("     - class0/class3 的 margin 恶化被大类掩盖")
    print("  3. T-CST 用边缘 MMD 优化，无法保证类别条件对齐")

    # 断言：边缘 MMD 在 S1 和 S2 两个方向都出错
    assert mmd_1 > cc_1 * 5, "S1: 边缘 MMD 应远大于类条件 MMD（比例污染）"
    assert cc_2 > mmd_2 * 3, "S2: 类条件 MMD 应远大于边缘 MMD（条件偏移被低估）"
    print("\n  ✅ 断言通过：边缘 MMD 在两个方向都偏离条件分布真实差异")


if __name__ == '__main__':
    main()
