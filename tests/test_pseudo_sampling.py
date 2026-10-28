import unittest

import numpy as np

from core.pseudo_sampling import sample_pseudo_indices
from core.pseudo_sampling import select_shape_rescue_indices


class PseudoSamplingTests(unittest.TestCase):
    def test_sqrt_sampling_moderates_tiny_classes_without_equalizing_them(self):
        labels = np.array([0] * 200 + [1] * 200 + [2] * 17)
        indices = sample_pseudo_indices(
            labels, 10000, np.random.default_rng(7))
        counts = np.bincount(labels[indices], minlength=3)
        self.assertLess(counts[2] / counts.sum(), 0.16)
        self.assertGreater(counts[2] / counts.sum(), 0.10)

    def test_spectral_rescue_requires_shape_margin_and_one_supported_view(self):
        shape = np.array([[0.08, 0.84, 0.08], [0.10, 0.80, 0.10], [0.15, 0.70, 0.15]])
        views = np.array([
            [[0.10, 0.80, 0.10], [0.80, 0.10, 0.10], [0.10, 0.80, 0.10]],
            [[0.20, 0.60, 0.20], [0.20, 0.20, 0.60], [0.34, 0.33, 0.33]],
        ])
        selected = select_shape_rescue_indices(
            shape, views, class_id=1, min_shape_confidence=0.75,
            min_shape_margin=0.20, min_view_confidence=0.55, limit=8)
        self.assertEqual(selected.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
