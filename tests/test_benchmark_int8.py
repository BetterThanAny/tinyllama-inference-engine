from __future__ import annotations

import unittest

from scripts.benchmark_int8 import acceptance, validate_contexts, validate_result


def summary(throughput: float, memory: int) -> dict[str, float | int]:
    return {
        "decode_tokens_per_second_median": throughput,
        "engine_peak_device_bytes": memory,
    }


class BenchmarkInt8Tests(unittest.TestCase):
    def test_comparison_only_does_not_weaken_default_context_gate(self) -> None:
        validate_contexts([128], comparison_only=True)
        validate_contexts([128, 4096], comparison_only=False)
        with self.assertRaises(ValueError):
            validate_contexts([128], comparison_only=False)

    def test_accepts_memory_reduction_when_throughput_does_not_improve(self) -> None:
        result = acceptance(summary(100.0, 1000), summary(80.0, 700))
        self.assertFalse(result["throughput_threshold_passed"])
        self.assertTrue(result["memory_threshold_passed"])
        self.assertTrue(result["passed"])

    def test_rejects_allocation_count_drift(self) -> None:
        result = {
            "gpu": "NVIDIA GeForce RTX 3080 Laptop GPU",
            "dtype": "int8_weight_only",
            "context": 128,
            "output_tokens": 32,
            "warmup": 3,
            "sample_count": 1,
            "batch": 1,
            "model_source_sha256": "a" * 64,
            "model_int8_sha256": "b" * 64,
            "model_weight_sha256": "b" * 64,
            "samples": [{"ttft_ms": 1.0}],
            "device_allocation_count_before_workload": 10,
            "device_allocation_count_after_workload": 11,
            "kv_cache_bytes": 1024,
        }
        with self.assertRaisesRegex(ValueError, "changed device allocation count"):
            validate_result(
                result,
                "int8_weight_only",
                128,
                32,
                3,
                1,
                {"source": "a" * 64, "int8_weight_only": "b" * 64, "float16": "c" * 64},
            )


if __name__ == "__main__":
    unittest.main()
