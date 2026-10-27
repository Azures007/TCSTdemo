"""v8.1.0 P0/P1/P2 单元测试：覆盖 legacy 行为、增强分支、checkpoint、空间隔离、标签一致性、身份映射、QP 风险控制。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))

import numpy as np
import torch
import torch.nn as nn

from core.spectral_transform import SmoothSpectralTransform, ClassSmoothSpectralTransform
from core.transport_solver import solve_margin_qp
from core.task_geometry import select_constraint_samples
from core.risk_controller import _check_spectral_plausibility


def _make_dummy_model(bands, num_classes):
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(num_classes, bands) * 0.5)
            self.bias = nn.Parameter(torch.zeros(num_classes))
        def forward(self, x):
            b, c, h, w = x.shape
            center = x[:, :, h//2, w//2]
            lg = center @ self.weight.t() + self.bias
            class _O: pass
            o = _O(); o.logits = lg; o.features = center
            return o
    return _M()


# ============ T1: legacy 行为（默认配置 = v8.0.9） ============
def test_legacy_default_config():
    """默认配置下 ckpt_mode=A，use_warmup_augmentation=False。"""
    import yaml
    for ds in ['houston', 'hyrank', 'shanghai']:
        p = ROOT / f'configs/hou_H6_ls05.yaml' if ds == 'houston' else \
            (ROOT / f'configs/hy_Y2_warm40.yaml' if ds == 'hyrank' else ROOT / f'configs/sh_V2_lr1e4.yaml')
        c = yaml.safe_load(p.read_text())
        ckpt = c['training'].get('ckpt_mode', 'A')
        assert ckpt == 'A', f"{ds}: ckpt_mode={ckpt}, expected A"
        aug = c['training'].get('use_warmup_augmentation', False)
        assert aug is False, f"{ds}: use_warmup_augmentation={aug}, expected False"
    print("  ✓ T1: legacy default (ckpt_mode=A, aug=False)")


# ============ T2: 身份映射（phi=0 时 transform 是恒等） ============
def test_identity_transform():
    T = SmoothSpectralTransform(48, K=8, rho_a=0.1, rho_b=0.1)
    x = torch.randn(4, 48, 11, 11)
    phi = torch.zeros(T.dim)
    y = T(x, phi)
    assert torch.allclose(x, y, atol=1e-6), f"identity failed: max diff {(x-y).abs().max():.6f}"
    print("  ✓ T2: identity transform (phi=0)")


# ============ T3: 类条件变换（labels 非空时走对应类） ============
def test_class_conditional():
    T = ClassSmoothSpectralTransform(48, num_classes=3, K=8, rho_a=0.1, rho_b=0.1)
    x = torch.randn(3, 48, 11, 11)
    phi = torch.zeros(T.dim)
    y = T(x, phi, labels=torch.tensor([0, 1, 2]))
    assert torch.allclose(x, y, atol=1e-6), "identity (class-cond) failed"
    # 非零 phi：class0 和 class1 应该有不同的变换
    phi2 = torch.zeros(T.dim)
    phi2[0] = 0.5  # 只改 class0 的第一个 alpha
    y2 = T(x, phi2, labels=torch.tensor([0, 0, 0]))
    assert not torch.allclose(x[0], y2[0], atol=1e-3), "class-conditional transform should change class0"
    print("  ✓ T3: class-conditional transform (labels 生效)")


# ============ T4: 空间隔离（strict_mixed_isolation） ============
def test_strict_mixed_isolation():
    from core.trainer_v8 import _spatial_block_split
    rng = np.random.default_rng(0)
    gt = np.zeros((200, 400), dtype=np.int64)
    for c in [1, 2, 3]:
        for _ in range(500):
            r = rng.integers(0, 200); col = rng.integers(0, 400)
            gt[r, col] = c
    masks_a, _, _ = _spatial_block_split(
        gt, train_ratio=1.0, constraint_per_class=12, calibration_per_class=12,
        val_frac_from_train=0.10, seed=42, grid_size=10,
        per_class_train_cap=150, per_class_val_target=150, patch_size=11,
        strict_mixed_isolation=False)
    masks_b, _, _ = _spatial_block_split(
        gt, train_ratio=1.0, constraint_per_class=12, calibration_per_class=12,
        val_frac_from_train=0.10, seed=42, grid_size=10,
        per_class_train_cap=150, per_class_val_target=150, patch_size=11,
        strict_mixed_isolation=True)
    # strict 模式下 val 数量应 ≤ loose 模式
    n_val_a = int(masks_a['val'].sum())
    n_val_b = int(masks_b['val'].sum())
    assert n_val_b <= n_val_a, f"strict val ({n_val_b}) > loose val ({n_val_a})"
    print(f"  ✓ T4: strict_mixed_isolation (loose val={n_val_a}, strict val={n_val_b})")


# ============ T5: 标签一致性（np.bincount 类数） ============
def test_label_consistency():
    from core.data import load_scene
    DATA = ROOT.parent / 'v7.0/PTMDA-Net_v7_reopt/Datasets'
    if not DATA.exists():
        print("  ⏭️ T5: skipped (data not available)")
        return
    for ds, (img, gt, nc) in {
        'Houston': ('Houston/Houston13.mat', 'Houston/Houston13_7gt.mat', 7),
        'HyRANK': ('HyRANK/Dioni.mat', 'HyRANK/Dioni_12gt.mat', 12),
        'S→H': ('Shanghai_Hangzhou/Shanghai-Hangzhou/Shanghai.mat',
                'Shanghai_Hangzhou/Shanghai-Hangzhou/Shanghai_3gt.mat', 3),
    }.items():
        _, gt_arr = load_scene(str(DATA / img), str(DATA / gt), None, nc, verbose=False)
        labels = np.unique(gt_arr)
        nz = labels[labels > 0]
        assert len(nz) == nc, f"{ds}: {len(nz)} classes but nc={nc}"
    print("  ✓ T5: label consistency (nc matches unique non-zero labels)")


# ============ T6: QP 风险控制（J==0 时返回零方向） ============
def test_qp_zero_constraints():
    g_D = np.array([1.0, 1.0], dtype=np.float32)
    out = solve_margin_qp(g_D, np.zeros((0, 2), np.float32),
                          np.zeros((0,), np.float32), np.zeros((0,), np.float32))
    assert out['d_norm'] == 0.0, f"J=0 should return zero direction, got {out['d_norm']}"
    assert out['status'] == 'no_constraints_rejected', f"unexpected status {out['status']}"
    print("  ✓ T6: QP J=0 returns zero direction (reject)")


# ============ T7: 光谱合理性阈值（默认参数） ============
def test_spectral_plausibility():
    ok_diag = {
        'gain_min': 0.95, 'gain_max': 1.05,
        'offset_min': -0.05, 'offset_max': 0.05,
        'gain_tv': 0.1, 'offset_tv': 0.05,
        'relative_spectral_change': 0.1,
    }
    ok, reason = _check_spectral_plausibility(ok_diag, 0.5, 0.5)
    assert ok, f"ok diag rejected: {reason}"

    bad_diag = dict(ok_diag)
    bad_diag['gain_tv'] = 0.5  # > gain_tv_max 0.36
    ok, reason = _check_spectral_plausibility(bad_diag, 0.5, 0.5)
    assert not ok, "bad gain_tv should be rejected"

    bad_diag2 = dict(ok_diag)
    bad_diag2['relative_spectral_change'] = 0.2  # > 0.15
    ok, reason = _check_spectral_plausibility(bad_diag2, 0.5, 0.5)
    assert not ok, "bad rel_change should be rejected"
    print("  ✓ T7: spectral plausibility thresholds")


# ============ T8: 奇数约束样本（n=1 时选 1 个不是 0） ============
def test_odd_constraint_samples():
    """B2 验证：n=1 时选中 1 个（不是 0）"""
    class OneHotModel(nn.Module):
        def __init__(self, bands=8, num_classes=3):
            super().__init__()
            w = torch.zeros(num_classes, bands)
            for c in range(num_classes):
                w[c, c] = 5.0
            self.weight = nn.Parameter(w)
            self.bias = nn.Parameter(torch.zeros(num_classes))
        def forward(self, x):
            b, c, h, w = x.shape
            center = x[:, :, h//2, w//2]
            lg = center @ self.weight.t() + self.bias
            class _O: pass
            o = _O(); o.logits = lg; o.features = center
            return o

    model = OneHotModel(bands=8, num_classes=3)
    labels = np.array([0]*1 + [1]*5 + [2]*4, dtype=np.int64)
    patches = np.random.randn(10, 8, 5, 5).astype(np.float32) * 0.01
    for i, c in enumerate(labels):
        patches[i, c, 2, 2] = 10.0

    # 先确认模型预测正确
    with torch.no_grad():
        pred = model(torch.from_numpy(patches)).logits.argmax(1).numpy()
    assert (pred == labels).all(), "dummy model 没预测对"

    idx, y, k1, k2 = select_constraint_samples(
        model, patches, labels, torch.device('cpu'), per_class=2, top_competitors=2)

    class0_count = int((y == 0).sum())
    class1_count = int((y == 1).sum())
    class2_count = int((y == 2).sum())
    assert class0_count >= 1, f"class0 (n=1) 应选 1 个，实际 {class0_count}"
    assert class1_count == 2, f"class1 (n=5, per=2) 应选 2 个，实际 {class1_count}"
    assert class2_count == 2, f"class2 (n=4, per=2) 应选 2 个，实际 {class2_count}"
    print(f"  \u2713 T8: n=1->{class0_count}, n=5->{class1_count}, n=4->{class2_count}")

def main():
    print("=" * 70)
    print("  v8.1.0 P0/P1/P2 单元测试")
    print("=" * 70)
    tests = [
        test_legacy_default_config,
        test_identity_transform,
        test_class_conditional,
        test_strict_mixed_isolation,
        test_label_consistency,
        test_qp_zero_constraints,
        test_spectral_plausibility,
        test_odd_constraint_samples,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print("=" * 70)
    print(f"  {len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
    print("  ✅ ALL TESTS PASSED")


if __name__ == '__main__':
    main()
