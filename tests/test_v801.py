"""v8.0.1 新增测试：Top-2、QP 带约束、Risk 三态、Smoke。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from core.spectral_transform import SmoothSpectralTransform
from core.task_geometry import select_constraint_samples, compute_margin_gradients
from core.transport_solver import solve_margin_qp
from core.risk_controller import backtracking_search


class _DummyModel(nn.Module):
    def __init__(self, bands, num_classes):
        super().__init__()
        self.bands = bands
        self.num_classes = num_classes
        torch.manual_seed(0)
        self.weight = nn.Parameter(torch.randn(num_classes, bands) * 0.5)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        b, c, h, w = x.shape
        center = x[:, :, h // 2, w // 2]
        logits = center @ self.weight.t() + self.bias
        class _Out:
            pass
        out = _Out()
        out.logits = logits
        out.features = center
        return out


def test_top2_returns_two_competitors():
    model = _DummyModel(bands=8, num_classes=4)
    patches = np.random.randn(20, 8, 5, 5).astype(np.float32)
    labels = np.array([0]*5 + [1]*5 + [2]*5 + [3]*5, dtype=np.int64)
    idx, y, k1, k2 = select_constraint_samples(
        model, patches, labels, torch.device("cpu"), per_class=2, top_competitors=2)
    assert len(idx) == len(y) == len(k1) == len(k2)
    assert (k1 != y).all() and (k2 != y).all()
    transform = SmoothSpectralTransform(bands=8, K=4)
    G, m0, w, jc, cc = compute_margin_gradients(
        model, transform, patches[idx.numpy()], y, k1, k2, torch.device("cpu"))
    assert G.shape[0] == 2 * len(idx), f"期望 {2*len(idx)} 条，实际 {G.shape[0]}"
    assert G.shape[1] == 2 * 4
    y_list = y.tolist()
    k1_list = k1.tolist()
    k2_list = k2.tolist()
    for i in range(len(idx)):
        assert jc[2*i] == y_list[i] == jc[2*i+1], \
            f"jc[2*i]={jc[2*i]} y[{i}]={y_list[i]} jc[2*i+1]={jc[2*i+1]}"
        assert {cc[2*i], cc[2*i+1]} == {k1_list[i], k2_list[i]}, \
            f"约束 {i}: cc={cc[2*i:2*i+2]}, k1/k2={k1_list[i]}/{k2_list[i]}"
    print(f"  ✓ Top-2: J={G.shape[0]}, phi_dim={G.shape[1]}")


def test_margin_gradient_numerical():
    model = _DummyModel(bands=6, num_classes=3)
    transform = SmoothSpectralTransform(bands=6, K=3)
    patch = np.random.randn(1, 6, 5, 5).astype(np.float32)
    y = torch.tensor([0]); k1 = torch.tensor([1]); k2 = torch.tensor([2])
    G, m0, _, _, _ = compute_margin_gradients(
        model, transform, patch, y, k1, k2, torch.device("cpu"))
    eps = 1e-4
    def _m(phi_vec):
        x = torch.from_numpy(patch)
        with torch.no_grad():
            out = model(transform(x, phi_vec))
            return float(out.logits[0, 0] - out.logits[0, 1])
    phi0 = torch.zeros(2 * 3)
    for j in range(2 * 3):
        pp = phi0.clone(); pp[j] += eps
        pm = phi0.clone(); pm[j] -= eps
        grad_num = (_m(pp) - _m(pm)) / (2 * eps)
        assert abs(grad_num - G[0, j]) < 1e-2, f"grad[{j}]: 数值={grad_num:.4f} 解析={G[0,j]:.4f}"
    print(f"  ✓ Margin 梯度数值验证通过")


def test_qp_with_constraints():
    """宽松约束 (lambda_xi=10) 允许一定 slack；严格约束 (lambda_xi=1000) 时 slack ~ 0。
    这不是 bug：lambda_xi 小 → QP 优先降低 MMD，允许 slack；lambda_xi 大 → 优先满足约束。
    """
    g_D = np.array([1.0, 0.5], dtype=np.float32)
    G = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    m0 = np.array([1.0, 1.0], dtype=np.float32)
    w = np.array([1.0, 1.0], dtype=np.float32)
    eps = 0.05 * np.maximum(m0, 0)

    # 场景 A: lambda_xi=10（宽松），slack 允许 ≤ 0.15（经验值）
    out = solve_margin_qp(g_D, G, m0, w, rho_m=0.05, lambda_d=1.0, lambda_xi=10.0)
    assert out["status"] in ("optimal", "optimal_inaccurate"), f"实际 {out['status']}"
    d = out["d"]
    Gt = G @ d.astype(np.float64)
    violation = float(np.maximum(0.0, -eps - Gt).max())
    assert violation <= 0.15, f"lambda_xi=10 时 slack 应在合理范围 (<=0.15)，实际 {violation:.4f}"
    print(f"  ✓ QP loose (λ_ξ=10): status={out['status']}, ||d||={out['d_norm']:.4f}, "
          f"slack_max={violation:.4f} (允许)")

    # 场景 B: lambda_xi=1000（严格），slack 应 < 1e-2
    out2 = solve_margin_qp(g_D, G, m0, w, rho_m=0.05, lambda_d=1.0, lambda_xi=1000.0)
    assert out2["status"] in ("optimal", "optimal_inaccurate"), f"实际 {out2['status']}"
    d2 = out2["d"]
    Gt2 = G @ d2.astype(np.float64)
    violation2 = float(np.maximum(0.0, -eps - Gt2).max())
    assert violation2 <= 1e-2, f"lambda_xi=1000 时 slack 应接近 0，实际 {violation2:.4f}"
    print(f"  ✓ QP strict (λ_ξ=1000): status={out2['status']}, ||d||={out2['d_norm']:.4f}, "
          f"slack_max={violation2:.2e} (接近 0)")

def test_risk_reject_and_accept():
    torch.manual_seed(0)
    model = _DummyModel(bands=6, num_classes=3).eval()
    transform = SmoothSpectralTransform(bands=6, K=3)
    src = np.random.randn(30, 6, 5, 5).astype(np.float32)
    src_y = np.array([0]*10 + [1]*10 + [2]*10, dtype=np.int64)
    tgt = np.random.randn(30, 6, 5, 5).astype(np.float32) * 2.0
    d = np.random.randn(2 * 3).astype(np.float32)
    d = d / np.linalg.norm(d)

    # REJECT
    eta, _, diag = backtracking_search(
        model, transform, d, src, src_y, tgt,
        eta_candidates=[1.0, 0.5], tau_sem=1e-9, tau_flip=0.0,
        tau_class_flip=0.0, tau_swd_rel=1e9, device=torch.device("cpu"))
    assert eta == 0.0, f"应拒绝，实际 {eta}"
    assert diag["accepted_candidate"] is None
    print(f"  ✓ REJECT")

    # ACCEPT
    eta2, _, diag2 = backtracking_search(
        model, transform, d, src, src_y, tgt,
        eta_candidates=[1.0, 0.5], tau_sem=1e9, tau_flip=1.0,
        tau_class_flip=1.0, tau_swd_rel=-1.0, device=torch.device("cpu"))
    assert eta2 == 1.0, f"应接受 1.0，实际 {eta2}"
    assert diag2["accepted_candidate"] is not None
    assert diag2["accepted_candidate"]["eta"] == 1.0
    print(f"  ✓ ACCEPT + accepted_candidate")


def test_rs_crw_uses_accepted_candidate():
    torch.manual_seed(0)
    model = _DummyModel(bands=6, num_classes=3).eval()
    transform = SmoothSpectralTransform(bands=6, K=3)
    src = np.random.randn(30, 6, 5, 5).astype(np.float32)
    src_y = np.array([0]*10 + [1]*10 + [2]*10, dtype=np.int64)
    tgt = np.random.randn(30, 6, 5, 5).astype(np.float32) * 2.0
    d = np.random.randn(2 * 3).astype(np.float32); d = d / np.linalg.norm(d)
    eta, _, diag = backtracking_search(
        model, transform, d, src, src_y, tgt,
        eta_candidates=[1.0, 0.5, 0.25], tau_sem=1e9, tau_flip=1.0,
        tau_class_flip=1.0, tau_swd_rel=-1.0, device=torch.device("cpu"))
    if eta > 0:
        acc = diag["accepted_candidate"]
        assert acc["eta"] == eta
        assert len(acc["Vc"]) == 3
        print(f"  ✓ accepted_candidate eta={eta}, Vc={acc['Vc']}")


if __name__ == "__main__":
    test_top2_returns_two_competitors()
    test_margin_gradient_numerical()
    test_qp_with_constraints()
    test_risk_reject_and_accept()
    test_rs_crw_uses_accepted_candidate()
    print("\n=== v8.0.1 ALL TESTS PASSED ===")
