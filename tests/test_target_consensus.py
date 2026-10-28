import unittest

import numpy as np
import torch

import core.discrepancy as discrepancy
import scripts.train as train
from core.spectral_transform import SmoothSpectralTransform


class TargetConsensusTests(unittest.TestCase):
    def test_target_consensus_statistics_is_available(self):
        self.assertTrue(callable(getattr(discrepancy, "target_consensus_statistics", None)))

    def test_train_entry_supports_seed_and_save_directory_overrides(self):
        self.assertTrue(callable(getattr(train, "apply_cli_overrides", None)))

        cfg = {"output": {"seed": 1237, "save_dir": "results/default"}}
        train.apply_cli_overrides(cfg, seed=42, save_dir="results/seed42")
        self.assertEqual(cfg["output"], {"seed": 42, "save_dir": "results/seed42"})

    def test_target_consensus_keeps_two_view_agreement(self):
        context = torch.tensor([[8.0, 0.0], [0.0, 8.0], [2.0, 0.0]])
        spectral = torch.tensor([[7.0, 0.0], [8.0, 0.0], [1.5, 0.0]])
        fused = torch.tensor([[9.0, 0.0], [8.0, 0.0], [1.0, 0.0]])

        probs, weights, agreement = discrepancy.target_consensus_statistics(
            context, spectral, fused, torch.tensor([1.0, 0.7, 0.3]))

        self.assertEqual(agreement.tolist(), [True, True, True])
        self.assertGreater(float(weights[1]), 0.0)
        self.assertGreater(float(weights[0]), float(weights[2]))
        torch.testing.assert_close(probs.sum(dim=1), torch.ones(3))

    def test_target_consensus_remains_finite_for_extreme_logits(self):
        context = torch.tensor([[1000.0, -1000.0]])
        spectral = torch.tensor([[-1000.0, 1000.0]])
        fused = context.clone()

        probs, weights, agreement = discrepancy.target_consensus_statistics(
            context, spectral, fused, torch.ones(1))

        self.assertTrue(torch.isfinite(probs).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertFalse(bool(agreement[0]))
        self.assertEqual(float(weights[0]), 0.0)

    def test_spectral_shape_posteriors_are_normalized_and_class_sensitive(self):
        source = torch.zeros(4, 4, 3, 3)
        source[0:2, :, :, :] = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 4, 1, 1)
        source[2:4, :, :, :] = torch.tensor([4.0, 3.0, 2.0, 1.0]).view(1, 4, 1, 1)
        labels = torch.tensor([0, 0, 1, 1])
        centers = discrepancy.build_spectral_shape_centroids(source, labels, 2)
        target = torch.cat([source[:1], source[2:3]], dim=0)
        probs = discrepancy.spectral_shape_posteriors(target, centers, temperature=0.05)
        self.assertEqual(tuple(probs.shape), (2, 2))
        torch.testing.assert_close(probs.sum(dim=1), torch.ones(2))
        self.assertGreater(float(probs[0, 0]), float(probs[0, 1]))
        self.assertGreater(float(probs[1, 1]), float(probs[1, 0]))

    def test_uniform_target_weights_preserve_original_mmd(self):
        torch.manual_seed(3)
        source = torch.randn(8, 12)
        target = torch.randn(7, 12)
        torch.testing.assert_close(
            discrepancy.mk_mmd2(source, target),
            discrepancy.mk_mmd2(source, target, y_weights=torch.ones(7)))

    def test_class_conditional_mmd_falls_back_when_support_is_insufficient(self):
        torch.manual_seed(7)
        transform = SmoothSpectralTransform(bands=4, K=4)
        source = torch.randn(12, 4, 3, 3)
        labels = torch.tensor([0] * 6 + [1] * 6)
        target = torch.randn(8, 4, 3, 3)
        descriptor = discrepancy.spectral_descriptor(target)
        probabilities = torch.full((8, 2), 0.5)
        weights = torch.ones(8)

        fallback_grad, fallback_loss = discrepancy.domain_gradient_mmd(
            transform, source, descriptor, source_labels=labels,
            target_class_probs=probabilities, target_weights=weights,
            min_effective_samples=20)
        marginal_grad, marginal_loss = discrepancy.domain_gradient_mmd(
            transform, source, descriptor, source_labels=labels,
            target_weights=weights)

        torch.testing.assert_close(fallback_grad, marginal_grad)
        self.assertAlmostEqual(fallback_loss, marginal_loss, places=6)
        self.assertTrue(np.isfinite(fallback_loss))

    def test_supported_class_conditional_mmd_has_finite_gradient(self):
        torch.manual_seed(17)
        transform = SmoothSpectralTransform(bands=4, K=4)
        source = torch.randn(16, 4, 3, 3)
        labels = torch.tensor([0] * 8 + [1] * 8)
        target = torch.randn(8, 4, 3, 3)
        probs = torch.tensor([[0.95, 0.05]] * 4 + [[0.05, 0.95]] * 4)

        grad, loss = discrepancy.domain_gradient_mmd(
            transform, source, discrepancy.spectral_descriptor(target),
            source_labels=labels, target_class_probs=probs,
            target_weights=torch.ones(8), min_effective_samples=3)

        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.norm()), 0.0)
        self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
