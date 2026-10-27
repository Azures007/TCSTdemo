"""Reproducible synthetic tests for the independent HFCST-v9 implementation."""
from __future__ import annotations
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from core.model_hfcst import HFCSTModel
from core.spectral_transform_hfcst import HierarchicalSpectralTransform, build_dct_basis
from core.discrepancy_hfcst import weighted_mmd, class_transport_from_embeddings, spectral_descriptor
from core.transport_solver_hfcst import solve_margin_qp
from core.risk_controller_hfcst import candidate_acceptance
from core.data_hfcst import global_spatial_split, audit_no_cross_split_patch_overlap, load_cube
from core.trainer_hfcst import (capture_transaction_state, restore_transaction_state,
                                run_hfcst, _teacher_assignments, _target_cache,
                                _transport_tensor, _train_epochs)


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_01_identity_transform():
    t = HierarchicalSpectralTransform(12, 3, K=4)
    x = torch.randn(5, 12, 7, 7)
    assert torch.allclose(t(x, labels=torch.tensor([0, 1, 2, 1, 0])), x, atol=1e-6)
    assert torch.allclose(t(x), x, atol=1e-6)


def test_02_dct_orthogonality():
    U = build_dct_basis(32, 8)
    assert torch.allclose(U.T @ U, torch.eye(8), atol=1e-5)


def test_03_shared_transform_degeneracy():
    t = HierarchicalSpectralTransform(12, 3, K=4)
    with torch.no_grad():
        t.phi_shared.copy_(torch.linspace(-0.3, 0.3, t.dim))
        t.delta_phi.zero_()
    x = torch.randn(3, 12, 5, 5)
    y = t(x, labels=torch.tensor([0, 1, 2]))
    assert torch.allclose(y, t(x), atol=1e-6)


def test_04_class_routing():
    t = HierarchicalSpectralTransform(12, 3, K=4)
    with torch.no_grad():
        t.delta_phi[0, 0] = 0.8
        t.delta_phi[1, 0] = -0.8
    x = torch.ones(2, 12, 5, 5)
    y = t(x, labels=torch.tensor([0, 1]))
    assert not torch.allclose(y[0], y[1])


def test_05_unreliable_class_residual_is_zero():
    t = HierarchicalSpectralTransform(12, 3, K=4)
    with torch.no_grad(): t.delta_phi.fill_(0.2)
    t.set_supported_classes([True, False, True])
    assert torch.count_nonzero(t.delta_phi[1]).item() == 0
    x = torch.randn(3, 12, 5, 5)
    y = t(x, labels=torch.tensor([1, 1, 1]))
    assert torch.allclose(y, t(x), atol=1e-6)


def test_06_weighted_mmd_normalization_and_identity():
    x = torch.randn(12, 6)
    w = torch.arange(1, 13, dtype=torch.float32)
    d = weighted_mmd(x, x.clone(), target_weights=w, source_weights=w)
    assert torch.isfinite(d) and float(d) < 1e-6
    y = torch.randn(9, 6)
    d2 = weighted_mmd(x, y, target_weights=torch.ones(9) * 2)
    assert torch.isfinite(d2) and float(d2) >= 0


def test_07_transport_gradients_are_finite():
    torch.manual_seed(4)
    model = HFCSTModel(8, 3, feature_dim=16)
    t = HierarchicalSpectralTransform(8, 3, K=3)
    sx = torch.randn(12, 8, 5, 5)
    sy = torch.tensor([0]*4 + [1]*4 + [2]*4)
    tx = torch.randn(15, 8, 5, 5) + 0.3
    with torch.no_grad():
        tout = model(tx)
        td = spectral_descriptor(tx)
    q = torch.softmax(torch.randn(15, 3), dim=1)
    r = torch.ones(15)
    xt = t(sx, labels=sy); sout = model(xt)
    sd = spectral_descriptor(xt)
    loss, diag = class_transport_from_embeddings(sd, sout.features, sy, td, tout.features,
                                                  q, r, 3, min_effective=1, min_source=2,
                                                  residual_penalty=0.01*t.delta_phi.square().sum())
    grads = torch.autograd.grad(loss, (t.phi_shared, t.delta_phi), allow_unused=True)
    assert torch.isfinite(loss)
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert diag["n_valid_classes"] >= 1


def test_08_qp_hard_margin_and_trust_region():
    g = np.array([1.0, 0.5, -0.2])
    G = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    margins = np.array([1.0, 0.5, 0.25])
    out = solve_margin_qp(g, G, margins, rho_m=0.05, lambda_d=0.1, trust_radius=0.2)
    assert out["status"] == "optimal", out
    d = out["d"]
    assert np.linalg.norm(d) <= 0.2 + 1e-5
    assert np.all(G @ d >= -0.05*np.maximum(margins, 0) - 1e-4)


def test_09_no_supported_class_is_safe():
    sd = torch.randn(6, 8); sf = torch.randn(6, 4); sy = torch.tensor([0, 0, 1, 1, 2, 2])
    td = torch.randn(7, 8); tf = torch.randn(7, 4)
    q = torch.zeros(7, 3); r = torch.zeros(7)
    loss, diag = class_transport_from_embeddings(sd, sf, sy, td, tf, q, r, 3,
                                                  min_effective=2, min_source=2)
    assert torch.isfinite(loss) and abs(float(loss)) < 1e-8
    assert diag["n_valid_classes"] == 0
    qp = solve_margin_qp(np.ones(3), np.zeros((0, 3)), np.zeros(0))
    assert qp["status"] == "no_constraints_rejected" and qp["d_norm"] == 0


def test_10_candidate_rejected_without_transport_improvement():
    risk = {"available": True, "new_error_rate": 0.0, "worst_class_new_error": 0.0, "margin_loss": 0.0}
    out = candidate_acceptance(1.0, 1.0, [0.0, 0.0, 0.0, 0.0, 0.0], risk, True, 0.0,
                               {"minimum_bootstrap_replicates": 5, "minimum_transport_improvement": 1e-5})
    assert not out["accepted"] and "transport_not_improved" in out["reasons"]


def test_11_source_risk_rejects_candidate():
    risk = {"available": True, "new_error_rate": 0.5, "worst_class_new_error": 0.8, "margin_loss": 0.3}
    out = candidate_acceptance(1.0, 0.5, [0.5]*10, risk, True, 0.0,
                               {"minimum_bootstrap_replicates": 5, "max_new_error_rate": 0.02,
                                "max_class_new_error_rate": 0.1, "max_margin_loss": 0.2})
    assert not out["accepted"]
    assert "source_new_error_budget_exceeded" in out["reasons"]


def test_12_full_transaction_rollback_restores_all_states():
    torch.manual_seed(0)
    model = HFCSTModel(8, 3, feature_dim=16)
    teacher = copy.deepcopy(model)
    transform = HierarchicalSpectralTransform(8, 3, K=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    x = torch.randn(4, 8, 5, 5); y = torch.tensor([0, 1, 2, 0])
    optimizer.zero_grad(); F.cross_entropy(model(x).logits, y).backward(); optimizer.step()
    snapshot = capture_transaction_state(model, optimizer, teacher, transform, scheduler, 2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters(): p.add_(torch.randn_like(p))
        teacher.context_head.bias.add_(2)
        transform.phi_shared.add_(1)
        transform.delta_phi.add_(1)
    optimizer.param_groups[0]["lr"] = 0.2
    committed = restore_transaction_state(snapshot, model, optimizer, teacher, transform, scheduler)
    assert committed == 2
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert torch.allclose(transform.phi_shared, torch.zeros_like(transform.phi_shared))
    assert torch.allclose(transform.delta_phi, torch.zeros_like(transform.delta_phi))
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_13_global_spatial_split_has_no_patch_overlap():
    rng = _rng(7)
    gt = rng.integers(1, 4, size=(64, 64), dtype=np.int64)
    valid = np.ones_like(gt, dtype=bool)
    coords, _, _ = global_spatial_split(gt, valid, 3, patch_size=5, grid_size=4, seed=9,
                                        train_cap=0, constraint_cap=0, calibration_cap=0, validation_cap=0)
    audit = audit_no_cross_split_patch_overlap(coords, 5)
    assert audit["no_cross_split_overlap"], audit
    assert all(len(coords[k]) > 0 for k in ("train", "constraint", "calibration", "validation"))


def test_14_image_loader_does_not_open_target_gt():
    # A missing GT path is intentionally irrelevant to the image-only loader.
    tmp = Path(__file__).resolve().parent / "_tmp_hfcst_loader.mat"
    try:
        sio.savemat(tmp, {"cube": np.random.randn(12, 13, 8).astype(np.float32)})
        cube, valid, audit = load_cube(tmp, verbose=False)
        assert cube.shape == (12, 13, 8) and valid.shape == (12, 13)
        assert not (tmp.parent / "definitely_missing_gt.mat").exists()
    finally:
        if tmp.exists(): tmp.unlink()


def test_15_end_to_end_synthetic_training_and_target_label_isolation(tmp_path, monkeypatch):
    rng = _rng(101)
    H = W = 48; bands = 8; classes = 3
    src = rng.normal(size=(H, W, bands)).astype(np.float32)
    tgt = (rng.normal(size=(H, W, bands)) + 0.2).astype(np.float32)
    gt = rng.integers(1, classes + 1, size=(H, W), dtype=np.int16)
    tgt_gt = rng.integers(1, classes + 1, size=(H, W), dtype=np.int16)
    sio.savemat(tmp_path / "src.mat", {"cube": src})
    sio.savemat(tmp_path / "src_gt.mat", {"gt": gt})
    sio.savemat(tmp_path / "tgt.mat", {"cube": tgt})
    sio.savemat(tmp_path / "tgt_gt.mat", {"gt": tgt_gt})
    import core.trainer_hfcst as trainer
    real_load_gt = trainer.load_gt
    calls = []
    def checked_load_gt(path, expected_shape=None):
        p = Path(path)
        if p.name == "tgt_gt.mat":
            assert (tmp_path / "out" / "final.pt").exists(), "target GT accessed before final checkpoint save"
            calls.append("target_gt_after_checkpoint")
        return real_load_gt(path, expected_shape)
    monkeypatch.setattr(trainer, "load_gt", checked_load_gt)
    cfg = {
        "dataset": {"name": "synthetic", "source_name": "S", "target_name": "T",
                    "source_image": str(tmp_path/"src.mat"), "source_gt": str(tmp_path/"src_gt.mat"),
                    "target_image": str(tmp_path/"tgt.mat"), "target_gt": str(tmp_path/"tgt_gt.mat"),
                    "num_classes": 3, "expected_bands": {"source": None, "target": None}, "band_mapping": None},
        "training": {"patch_size": 5, "spatial_grid_size": 4, "epochs_warmup": 1,
                     "epochs_per_cycle": 1, "batch_size": 6, "lr": 0.0005, "weight_decay": 0.0,
                     "feature_dim": 16, "max_gate": 0.5, "per_class_train_cap": 6,
                     "per_class_constraint_cap": 3, "per_class_calibration_cap": 4,
                     "per_class_validation_cap": 4, "target_pool_size": 24,
                     "target_validation_pool_size": 18, "lambda_transport": 0.2,
                     "ema_decay": 0.9, "scale_quantile": 0.95},
        "hfcst": {"K": 3, "num_cycles": 1, "epochs_per_cycle": 1,
                  "lambda_spec": 1.0, "lambda_feat": 1.0, "lambda_residual": 0.01,
                  "min_effective_target_n": 2.0, "min_source_per_class": 2,
                  "transport_source_per_class": 4, "per_class_constraint_samples": 2,
                  "shared_trust_radius": 0.1, "residual_trust_radius": 0.05,
                  "eta_candidates": [1.0, 0.5], "bootstrap_replicates": 2,
                  "minimum_bootstrap_replicates": 1, "bootstrap_confidence": 0.99,
                  "minimum_transport_improvement": 0.0, "minimum_relative_improvement": 0.0,
                  "max_new_error_rate": 1.0, "max_class_new_error_rate": 1.0,
                  "max_margin_loss": 100.0, "margin_tolerance": 100.0,
                  "max_target_pseudo_flip_rate": 1.0, "max_block_target_flip_rate": 1.0,
                  "min_block_view_agreement": 0.0, "gain_max_deviation": 1.0,
                  "offset_max_abs": 1.0, "gain_tv_max": 10.0, "offset_tv_max": 10.0,
                  "relative_change_max": 10.0, "post_transport_tolerance": 10.0},
        "output": {"seed": 5, "save_dir": str(tmp_path/"out")}}
    results = run_hfcst(cfg, torch.device("cpu"), config_path="synthetic-test")
    assert (tmp_path / "out" / "baseline.pt").exists()
    assert (tmp_path / "out" / "final.pt").exists()
    assert (tmp_path / "out" / "events.jsonl").exists()
    assert results["target_gt_used_for_selection"] is False
    assert results["committed_cycles"] == 1
    assert calls == ["target_gt_after_checkpoint"]
    events = [json.loads(line) for line in (tmp_path / "out" / "events.jsonl").read_text().splitlines()]
    assert any(e.get("event") == "qp" and e.get("phase") == "shared" for e in events)
    assert any(e.get("event") == "candidate" and e.get("phase") == "shared" for e in events)
    assert any(e.get("event") == "cycle_committed" for e in events)
    # The teacher soft-assignment routine returns only distributions/reliability, no GT interface.
    model = HFCSTModel(8, 3, feature_dim=16)
    patches = rng.normal(size=(10, 8, 5, 5)).astype(np.float32)
    soft = _teacher_assignments(model, patches, torch.device("cpu"), 3, seed=3)
    assert soft["q"].shape == (10, 3) and soft["r"].shape == (10,)
    assert np.allclose(soft["q"].sum(axis=1), 1.0, atol=1e-5)


def test_16_all_unsupported_classes_skip_training_after_baseline(tmp_path):
    rng = _rng(18); H=W=48; bands=8; classes=3
    for name, arr in [("src.mat", rng.normal(size=(H,W,bands)).astype(np.float32)),
                      ("tgt.mat", rng.normal(size=(H,W,bands)).astype(np.float32))]:
        sio.savemat(tmp_path/name, {"cube": arr})
    gt = rng.integers(1, classes+1, size=(H,W), dtype=np.int16)
    sio.savemat(tmp_path/"src_gt.mat", {"gt": gt})
    sio.savemat(tmp_path/"tgt_gt.mat", {"gt": gt})
    cfg = {"dataset": {"name":"fallback", "source_name":"S", "target_name":"T",
                        "source_image":str(tmp_path/"src.mat"), "source_gt":str(tmp_path/"src_gt.mat"),
                        "target_image":str(tmp_path/"tgt.mat"), "target_gt":str(tmp_path/"tgt_gt.mat"),
                        "num_classes":3, "expected_bands":{"source":None,"target":None}, "band_mapping":None},
           "training":{"patch_size":5,"spatial_grid_size":4,"epochs_warmup":1,"batch_size":6,
                       "lr":0.0005,"weight_decay":0.0,"feature_dim":16,"per_class_train_cap":6,
                       "per_class_constraint_cap":3,"per_class_calibration_cap":4,"per_class_validation_cap":4,
                       "target_pool_size":24,"target_validation_pool_size":18,"scale_quantile":0.95},
           "hfcst":{"K":3,"num_cycles":1,"min_effective_target_n":1e9,"min_source_per_class":2},
           "output":{"seed":6,"save_dir":str(tmp_path/"out")}}
    run_hfcst(cfg, torch.device("cpu"), config_path="synthetic-fallback")
    base = torch.load(tmp_path/"out"/"baseline.pt", map_location="cpu")
    final = torch.load(tmp_path/"out"/"final.pt", map_location="cpu")
    assert all(torch.equal(base["model"][k], final["model"][k]) for k in base["model"])
    events = [json.loads(line) for line in (tmp_path/"out"/"events.jsonl").read_text().splitlines()]
    assert any(e.get("event") == "cycle_rejected" and e.get("pure_source_ce_executed") is False for e in events)
    assert not any(e.get("event") == "cycle_committed" for e in events)
