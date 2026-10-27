import unittest

import numpy as np
import torch

from core.discrepancy import domain_gradient_mmd, mk_mmd2, spectral_descriptor
from core.spectral_transform import (
    HierarchicalSpectralTransform,
    class_residual_reliability,
)


class HierarchicalTransportTests(unittest.TestCase):
    def test_reliability_zeros_uncertain_and_unsupported_classes(self):
        probabilities = np.full((256, 3), 1 / 3, dtype=np.float32)
        reliability = class_residual_reliability(probabilities)
        np.testing.assert_array_equal(reliability, np.zeros(3, dtype=np.float32))

        probabilities = np.zeros((256, 3), dtype=np.float32)
        probabilities[:, 0] = 1.0
        reliability = class_residual_reliability(probabilities)
        self.assertEqual(reliability[0], 1.0)
        np.testing.assert_array_equal(reliability[1:], np.zeros(2, dtype=np.float32))

    def test_unreliable_class_residual_falls_back_to_global(self):
        transform = HierarchicalSpectralTransform(16, 3, K=8, residual_k=2)
        transform.class_reliability.copy_(torch.tensor([1.0, 0.0, 0.0]))
        phi = torch.zeros(transform.dim)
        phi[transform.global_dim + transform.num_classes * transform.residual_k] = 2.0
        x = torch.rand(3, 16, 5, 5)
        labels = torch.tensor([0, 1, 2])

        result = transform(x, phi, labels=labels)

        self.assertFalse(torch.allclose(result[0], x[0]))
        torch.testing.assert_close(result[1:], x[1:])

    def test_conditional_mmd_updates_only_reliable_class_residuals(self):
        torch.manual_seed(7)
        transform = HierarchicalSpectralTransform(8, 2, K=4, residual_k=2)
        transform.class_reliability.copy_(torch.tensor([1.0, 0.0]))
        source = torch.randn(16, 8, 3, 3)
        labels = torch.tensor([0] * 8 + [1] * 8)
        target = torch.randn(12, 8, 3, 3) + 0.4
        target_probs = torch.softmax(torch.randn(12, 2), dim=1)

        gradient, _ = domain_gradient_mmd(
            transform, source, spectral_descriptor(target),
            source_labels=labels, target_class_probs=target_probs)

        active = gradient[transform.global_dim:transform.global_dim + 2]
        inactive = gradient[transform.global_dim + 2:transform.global_dim + 4]
        self.assertGreater(float(active.norm()), 1e-7)
        torch.testing.assert_close(inactive, torch.zeros_like(inactive))

    def test_reliability_setter_accepts_probability_matrix(self):
        transform = HierarchicalSpectralTransform(8, 2, K=4, residual_k=2)
        probabilities = torch.zeros(256, 2)
        probabilities[:, 0] = 1.0

        reliability = transform.set_class_reliability(probabilities)

        self.assertEqual(float(reliability[0]), 1.0)
        self.assertEqual(float(reliability[1]), 0.0)

    def test_hierarchical_transform_starts_as_identity(self):
        transform = HierarchicalSpectralTransform(16, 3, K=8, residual_k=2)
        x = torch.rand(4, 16, 5, 5)
        result = transform(x, torch.zeros(transform.dim), labels=torch.tensor([0, 1, 2, 1]))
        torch.testing.assert_close(result, x)

    def test_uniform_target_weights_preserve_existing_mmd(self):
        torch.manual_seed(2)
        source = torch.randn(8, 12)
        target = torch.randn(7, 12)
        plain = mk_mmd2(source, target)
        weighted = mk_mmd2(source, target, y_weights=torch.ones(7))
        torch.testing.assert_close(plain, weighted)


if __name__ == "__main__":
    unittest.main()
