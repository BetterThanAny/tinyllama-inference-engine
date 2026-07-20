from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import benchmark_llamacpp, load_test


class M5RuntimeContractTests(unittest.TestCase):
    def test_stability_requires_terminal_request_accounting(self) -> None:
        valid = {
            "submitted_total": 100,
            "completed_total": 80,
            "cancelled_total": 20,
            "failed_total": 0,
        }
        self.assertTrue(load_test.validate_request_accounting(valid)["passed"])

        lost = {**valid, "completed_total": 79}
        self.assertFalse(load_test.validate_request_accounting(lost)["passed"])
        failed = {**valid, "completed_total": 79, "failed_total": 1}
        self.assertFalse(load_test.validate_request_accounting(failed)["passed"])

    def test_llama_checkout_must_match_pinned_clean_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = root / "build-cuda" / "bin" / "llama-server"
            server.parent.mkdir(parents=True)
            server.write_bytes(b"binary")
            with mock.patch.object(
                benchmark_llamacpp.subprocess,
                "run",
                side_effect=[
                    subprocess.CompletedProcess([], 0, stdout="a" * 40 + "\n", stderr=""),
                    subprocess.CompletedProcess([], 0, stdout="", stderr=""),
                ],
            ):
                self.assertEqual(
                    benchmark_llamacpp.validate_llama_checkout(server, "a" * 40), root.resolve()
                )

            with (
                mock.patch.object(
                    benchmark_llamacpp.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(
                        [], 0, stdout="b" * 40 + "\n", stderr=""
                    ),
                ),
                self.assertRaisesRegex(ValueError, "commit"),
            ):
                benchmark_llamacpp.validate_llama_checkout(server, "a" * 40)

    def test_llama_batch_requires_ten_finite_positive_samples(self) -> None:
        samples = [1.0] * 10
        self.assertEqual(benchmark_llamacpp.validate_batch_samples(samples, 10), samples)
        with self.assertRaisesRegex(ValueError, "10"):
            benchmark_llamacpp.validate_batch_samples([1.0], 10)
        with self.assertRaisesRegex(ValueError, "finite positive"):
            benchmark_llamacpp.validate_batch_samples([1.0] * 9 + [float("nan")], 10)


if __name__ == "__main__":
    unittest.main()
