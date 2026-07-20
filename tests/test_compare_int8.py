from __future__ import annotations

import unittest

import numpy as np

from scripts.compare_int8 import common_prefix, distribution_metrics


class CompareInt8Tests(unittest.TestCase):
    def test_distribution_metrics_detect_top1_and_finite_error(self) -> None:
        reference = np.asarray([1.0, 3.0, -2.0], dtype=np.float32)
        actual = np.asarray([1.1, 2.9, -1.8], dtype=np.float32)
        metrics = distribution_metrics(actual, reference)

        self.assertTrue(metrics["top1_matches"])
        self.assertGreater(float(metrics["cosine_similarity"]), 0.99)
        self.assertGreater(float(metrics["max_abs_error"]), 0.0)
        self.assertGreaterEqual(float(metrics["jensen_shannon_divergence"]), 0.0)

    def test_common_prefix_stops_at_first_mismatch(self) -> None:
        self.assertEqual(common_prefix([1, 2, 9, 4], [1, 2, 3, 4]), 2)

    def test_generation_gate_cannot_treat_a_prefix_as_exact(self) -> None:
        golden = [1, 2, 3, 4]
        degraded = [1, 2, 9, 9]
        self.assertEqual(common_prefix(degraded, golden), 2)
        self.assertNotEqual(degraded, golden)


if __name__ == "__main__":
    unittest.main()
