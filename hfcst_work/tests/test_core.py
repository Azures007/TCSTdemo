import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from core.spectral_transform import SmoothSpectralTransform, build_dct_basis
from core.discrepancy import (spectral_descriptor, mk_mmd2,
                              domain_gradient_mmd, sliced_wasserstein)
from core.transport_solver import solve_margin_qp
from core.model import DualReliabilityEncoder


def test_identity():
    T = SmoothSpectralTransform(48, K=8)
    x = torch.randn(4, 48, 9, 9)
    y = T(x, torch.zeros(16))
    assert torch.allclose(x, y, atol=1e-6)


def test_dct():
    U = build_dct_basis(48, 8)
    assert torch.allclose(U.t() @ U, torch.eye(8), atol=1e-4)


def test_descriptor():
    d = spectral_descriptor(torch.randn(5, 48, 9, 9))
    assert d.shape == (5, 4*48-3)


def test_grad():
    T = SmoothSpectralTransform(48, K=8)
    g, m = domain_gradient_mmd(T, torch.randn(8, 48, 9, 9),
                               spectral_descriptor(torch.randn(8, 48, 9, 9)))
    assert g.shape == (16,) and torch.isfinite(g).all() and m >= 0


def test_swd_zero():
    x = torch.randn(8, 4)
    assert abs(sliced_wasserstein(x, x, seed=0)) < 1e-6


def test_qp():
    g_D = np.array([1.0, 1.0], dtype=np.float32)
    out = solve_margin_qp(g_D, np.zeros((0, 2), np.float32),
                          np.zeros((0,), np.float32), np.zeros((0,), np.float32))
    assert out["directional_derivative"] <= 0


def test_backbone_forward():
    m = DualReliabilityEncoder(bands=48, num_classes=7,
                               projection_dim=64, projection_norm="batch")
    out = m(torch.randn(2, 48, 11, 11))
    assert out.logits.shape == (2, 7)


if __name__ == "__main__":
    test_identity(); test_dct(); test_descriptor()
    test_grad(); test_swd_zero(); test_qp(); test_backbone_forward()
    print("ALL TESTS PASSED")
