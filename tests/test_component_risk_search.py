import unittest

import numpy as np
import torch
from torch import nn

from core.discrepancy import domain_gradient_mmd, mk_mmd2, spectral_descriptor
from core.risk_controller import backtracking_search
from core.spectral_transform import HierarchicalSpectralTransform
from scripts.train import apply_cli_overrides


class ThresholdModel(nn.Module):
    def forward(self, x):
        score = x.mean(dim=(1, 2, 3)) - 0.95
        return type("Output", (), {"logits": torch.stack([score, -score], dim=1)})


class ComponentRiskSearchTests(unittest.TestCase):
    def test_cli_overrides_seed_and_output_directory(self):
        cfg = {"output": {"seed": 1237, "save_dir": "results/default"}}

        result = apply_cli_overrides(cfg, seed=42, save_dir="results/seed42")

        self.assertIs(result, cfg)
        self.assertEqual(result["output"]["seed"], 42)
        self.assertEqual(result["output"]["save_dir"], "results/seed42")

    def test_class_residual_only_changes_reliable_class(self):
        transform = HierarchicalSpectralTransform(
            bands=2, num_classes=2, K=2, residual_k=1,
            rho_a=0.1, rho_b=0.1)
        transform.class_reliability.copy_(torch.tensor([1.0, 0.0]))
        x = torch.ones(2, 2, 3, 3)
        labels = torch.tensor([0, 1])
        phi = torch.zeros(transform.dim)
        phi[transform.global_dim] = 2.0

        result = transform(x, phi, labels=labels)

        self.assertFalse(torch.equal(result[0], x[0]))
        torch.testing.assert_close(result[1], x[1])

    def test_rejected_class_residual_keeps_accepted_global_transform(self):
        transform = HierarchicalSpectralTransform(
            bands=2, num_classes=2, K=2, residual_k=1,
            rho_a=0.1, rho_b=0.1)
        transform.class_reliability.copy_(torch.tensor([1.0, 0.0]))
        model = ThresholdModel()
        source = np.ones((4, 2, 3, 3), dtype=np.float32)
        labels = np.zeros(4, dtype=np.int64)
        target = np.ones((4, 2, 3, 3), dtype=np.float32)
        base_phi = torch.zeros(transform.dim)
        base_phi[transform.K] = 0.5
        residual = np.zeros(transform.dim, dtype=np.float32)
        beta_class0 = transform.global_dim + 2 * transform.residual_k
        residual[beta_class0] = -8.0

        eta, result, diag = backtracking_search(
            model, transform, residual, source, labels, target,
            eta_candidates=[1.0], tau_sem=1.0, tau_flip=0.02,
            tau_class_flip=0.1, tau_swd_rel=0.01, device=torch.device("cpu"),
            base_phi=base_phi, risk_class=0,
            target_class_probs=torch.ones(4),
            domain_cal_patches=source, domain_cal_labels=labels)

        self.assertEqual(eta, 0.0)
        torch.testing.assert_close(result, base_phi)
        self.assertEqual(diag["best_rejected"]["risk_class"], 0)

    def test_missing_class_calibration_rejects_residual_without_losing_base(self):
        transform = HierarchicalSpectralTransform(
            bands=2, num_classes=2, K=2, residual_k=1,
            rho_a=0.1, rho_b=0.1)
        model = ThresholdModel()
        source = np.ones((4, 2, 3, 3), dtype=np.float32)
        labels = np.zeros(4, dtype=np.int64)
        target = np.ones((4, 2, 3, 3), dtype=np.float32)
        base_phi = torch.zeros(transform.dim)
        base_phi[transform.K] = 0.25
        residual = np.zeros(transform.dim, dtype=np.float32)
        residual[-1] = 1.0

        eta, result, diag = backtracking_search(
            model, transform, residual, source, labels, target,
            eta_candidates=[1.0], tau_sem=1.0, tau_flip=0.02,
            tau_class_flip=0.1, tau_swd_rel=0.01, device=torch.device("cpu"),
            base_phi=base_phi, risk_class=1,
            target_class_probs=torch.ones(4),
            domain_cal_patches=source, domain_cal_labels=labels)

        self.assertEqual(eta, 0.0)
        torch.testing.assert_close(result, base_phi)
        self.assertEqual(diag["skipped"], "no_source_calibration_support")

    def test_uniform_target_weights_match_unweighted_mmd(self):
        torch.manual_seed(11)
        source = torch.randn(8, 12)
        target = torch.randn(7, 12)
        torch.testing.assert_close(
            mk_mmd2(source, target),
            mk_mmd2(source, target, y_weights=torch.ones(7)))

    def test_class_conditioned_mmd_has_finite_class_residual_gradient(self):
        torch.manual_seed(7)
        transform = HierarchicalSpectralTransform(
            bands=4, num_classes=2, K=4, residual_k=2)
        transform.class_reliability.fill_(1.0)
        source = torch.randn(12, 4, 3, 3)
        labels = torch.tensor([0] * 6 + [1] * 6)
        target = torch.randn(10, 4, 3, 3)
        target_probs = torch.softmax(torch.randn(10, 2), dim=1)

        grad, loss = domain_gradient_mmd(
            transform, source, spectral_descriptor(target), source_labels=labels,
            target_class_probs=target_probs)

        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(float(grad[transform.global_dim:].norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
