from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts import compare_engines, load_test


class M5ToolTests(unittest.TestCase):
    def test_formal_stability_rejects_shortened_duration(self) -> None:
        load_test.validate_formal_workload(20, 1800)
        with self.assertRaisesRegex(ValueError, "duration 1800"):
            load_test.validate_formal_workload(20, 1)

    def test_memory_stability_uses_settled_windows(self) -> None:
        stable = load_test.validate_memory_stability([900.0] * 6 + [1000.0] * 24 + [1020.0] * 10)
        self.assertTrue(stable["passed"])
        self.assertGreater(stable["discarded_warmup_samples"], 0)
        growing = load_test.validate_memory_stability([900.0] * 6 + [1000.0] * 24 + [1100.0] * 10)
        self.assertFalse(growing["passed"])

    def test_response_metrics_reject_nan_and_missing_fields(self) -> None:
        metrics = {
            "metrics": {
                "queue_ms": 1.0,
                "ttft_ms": 2.0,
                "tpot_ms": 3.0,
                "output_tokens_per_second": 4.0,
            }
        }
        self.assertTrue(load_test.finite_response_metrics(metrics))
        metrics["metrics"]["ttft_ms"] = float("nan")
        self.assertFalse(load_test.finite_response_metrics(metrics))

    def test_cross_engine_rows_require_all_real_engines(self) -> None:
        rows = [
            {
                "engine": engine,
                "source_tree_sha256": "a" * 64,
                "status": "available",
                "context": 128,
                "output_tokens": 32,
                "warmup": 3,
                "samples": 10,
                "sampling": "greedy",
                "seed": 0,
                "ttft_ms_median": 1.0,
                "tpot_ms_median": 1.0,
                "output_tokens_per_second_median": 1.0,
                "peak_device_bytes": 1,
                "generated_tokens": list(range(32)),
                "software_thermal_slowdown_states": ["Not Active"],
                "thermal_clean": True,
            }
            for engine in compare_engines.REQUIRED_ENGINES
        ]
        compare_engines.validate_rows(rows)
        rows.pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            compare_engines.validate_rows(rows)

    def test_cross_engine_rows_reject_token_drift(self) -> None:
        rows = [
            {
                "engine": engine,
                "source_tree_sha256": "a" * 64,
                "status": "available",
                "context": 128,
                "output_tokens": 32,
                "warmup": 3,
                "samples": 10,
                "sampling": "greedy",
                "seed": 0,
                "ttft_ms_median": 1.0,
                "tpot_ms_median": 1.0,
                "output_tokens_per_second_median": 1.0,
                "peak_device_bytes": 1,
                "generated_tokens": list(range(32)),
                "software_thermal_slowdown_states": ["Not Active"],
                "thermal_clean": True,
            }
            for engine in compare_engines.REQUIRED_ENGINES
        ]
        rows[-1]["generated_tokens"] = [99, *range(1, 32)]
        with self.assertRaisesRegex(ValueError, "greedy tokens differ"):
            compare_engines.validate_rows(rows)

    def test_cross_engine_rows_reject_mixed_source_snapshots(self) -> None:
        rows = [
            {
                "engine": engine,
                "source_tree_sha256": "a" * 64,
                "status": "available",
                "context": 128,
                "output_tokens": 32,
                "warmup": 3,
                "samples": 10,
                "sampling": "greedy",
                "seed": 0,
                "ttft_ms_median": 1.0,
                "tpot_ms_median": 1.0,
                "output_tokens_per_second_median": 1.0,
                "peak_device_bytes": 1,
                "generated_tokens": list(range(32)),
                "software_thermal_slowdown_states": ["Not Active"],
                "thermal_clean": True,
            }
            for engine in compare_engines.REQUIRED_ENGINES
        ]
        rows[-1]["source_tree_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "mixed source snapshots"):
            compare_engines.validate_rows(rows)

    def test_cross_engine_rows_reject_non_hex_source_snapshot(self) -> None:
        rows = self._valid_cross_engine_rows()
        for row in rows:
            row["source_tree_sha256"] = "z" * 64
        with self.assertRaisesRegex(ValueError, "source snapshots"):
            compare_engines.validate_rows(rows)

    def test_cross_engine_rows_reject_non_finite_metrics(self) -> None:
        for value in (float("nan"), float("inf")):
            rows = self._valid_cross_engine_rows()
            rows[0]["ttft_ms_median"] = value
            with self.assertRaisesRegex(ValueError, "invalid ttft_ms_median"):
                compare_engines.validate_rows(rows)

    def test_cross_engine_rows_reject_empty_thermal_samples(self) -> None:
        rows = self._valid_cross_engine_rows()
        rows[0]["software_thermal_slowdown_states"] = []
        with self.assertRaisesRegex(ValueError, "invalid thermal metadata"):
            compare_engines.validate_rows(rows)

    @staticmethod
    def _valid_cross_engine_rows() -> list[dict[str, Any]]:
        return [
            {
                "engine": engine,
                "source_tree_sha256": "a" * 64,
                "status": "available",
                "context": 128,
                "output_tokens": 32,
                "warmup": 3,
                "samples": 10,
                "sampling": "greedy",
                "seed": 0,
                "ttft_ms_median": 1.0,
                "tpot_ms_median": 1.0,
                "output_tokens_per_second_median": 1.0,
                "peak_device_bytes": 1,
                "generated_tokens": list(range(32)),
                "software_thermal_slowdown_states": ["Not Active"],
                "thermal_clean": True,
            }
            for engine in compare_engines.REQUIRED_ENGINES
        ]

    def test_external_report_cannot_mark_unavailable_as_a_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(
                json.dumps({"engine": "llama_cpp", "status": "unavailable"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "missing a successful real-engine"):
                compare_engines.external_row(path, "llama_cpp")

    def test_cross_engine_rows_reject_shortened_sampling(self) -> None:
        rows = [
            {
                "engine": engine,
                "source_tree_sha256": "a" * 64,
                "status": "available",
                "context": 128,
                "output_tokens": 32,
                "warmup": 3,
                "samples": 10,
                "sampling": "greedy",
                "seed": 0,
                "ttft_ms_median": 1.0,
                "tpot_ms_median": 1.0,
                "output_tokens_per_second_median": 1.0,
                "peak_device_bytes": 1,
                "generated_tokens": list(range(32)),
                "software_thermal_slowdown_states": ["Not Active"],
                "thermal_clean": True,
            }
            for engine in compare_engines.REQUIRED_ENGINES
        ]
        rows[-1]["samples"] = 1
        with self.assertRaisesRegex(ValueError, "non-comparable workload"):
            compare_engines.validate_rows(rows)


if __name__ == "__main__":
    unittest.main()
