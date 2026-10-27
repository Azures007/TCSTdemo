"""修复 T8 测试：用 one-hot dummy model 保证 class0 预测正确"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))

import numpy as np
import torch
import torch.nn as nn
from core.task_geometry import select_constraint_samples


class OneHotModel(nn.Module):
    """weight = one-hot，bias = 0。输入 center 是 one-hot 时，logits = 对应类。"""
    def __init__(self, bands=8, num_classes=3):
        super().__init__()
        w = torch.zeros(num_classes, bands)
        for c in range(num_classes):
            w[c, c] = 5.0  # 大权重让 argmax 明确
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        b, c, h, w = x.shape
        center = x[:, :, h//2, w//2]
        lg = center @ self.weight.t() + self.bias
        class _O: pass
        o = _O(); o.logits = lg; o.features = center
        return o


def make_onehot_patches(labels, bands=8, ps=5):
    """每个样本的 center 是 one-hot，其余是噪声。"""
    n = len(labels)
    p = np.random.randn(n, bands, ps, ps).astype(np.float32) * 0.01
    for i, c in enumerate(labels):
        p[i, c, ps//2, ps//2] = 10.0  # center 是 one-hot（放大）
    return p


def test_odd_constraint_samples_v2():
    model = OneHotModel(bands=8, num_classes=3)
    # class0 only 1 sample, class1 5, class2 4
    labels = np.array([0]*1 + [1]*5 + [2]*4, dtype=np.int64)
    patches = make_onehot_patches(labels)

    # 先验证模型确实能预测对
    with torch.no_grad():
        lg = model(torch.from_numpy(patches)).logits
        pred = lg.argmax(1).numpy()
    correct = (pred == labels).sum()
    print(f"  模型预测正确: {correct}/{len(labels)}")
    assert correct == len(labels), f"dummy model 没预测对，测试无意义"

    idx, y, k1, k2 = select_constraint_samples(
        model, patches, labels, torch.device('cpu'), per_class=2, top_competitors=2)

    for c in [0, 1, 2]:
        n_c = int((y == c).sum())
        print(f"  class{c}: {n_c} 个约束样本（原 label 数={int((labels == c).sum())}）")

    # class0 只有 1 个样本，应该被选中 1 个（不是 0）
    class0_count = int((y == 0).sum())
    assert class0_count >= 1, f"class0 (n=1) 应被选中 1 个，实际 {class0_count}"

    # class1 有 5 个，per_class=2，应该选 2 个
    class1_count = int((y == 1).sum())
    assert class1_count == 2, f"class1 (n=5, per_class=2) 应选 2 个，实际 {class1_count}"

    # class2 有 4 个，per_class=2，应该选 2 个
    class2_count = int((y == 2).sum())
    assert class2_count == 2, f"class2 (n=4, per_class=2) 应选 2 个，实际 {class2_count}"

    # k1/k2 不能等于真实类
    for i in range(len(y)):
        assert int(k1[i]) != int(y[i]), f"约束 {i}: k1={k1[i]} 等于 true={y[i]}"
        assert int(k2[i]) != int(y[i]), f"约束 {i}: k2={k2[i]} 等于 true={y[i]}"

    print(f"  ✓ T8 v2: n=1 选中 {class0_count}，n=5 选中 {class1_count}，n=4 选中 {class2_count}")


if __name__ == '__main__':
    test_odd_constraint_samples_v2()
    print("✅ T8 v2 PASSED")
