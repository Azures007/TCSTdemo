import unittest

import numpy as np

from core.pseudo_sampling import sample_pseudo_indices


class PseudoSamplingTests(unittest.TestCase):
    def test_uniform_pool_sampling_does_not_equalize_tiny_classes(self):
        labels = np.array([0] * 200 + [1] * 200 + [2] * 17)
        indices = sample_pseudo_indices(
            labels, 10000, np.random.default_rng(7), mode="pool_uniform")
        counts = np.bincount(labels[indices], minlength=3)
        self.assertLess(counts[2] / counts.sum(), 0.06)
        self.assertGreater(counts[2] / counts.sum(), 0.02)


if __name__ == "__main__":
    unittest.main()
