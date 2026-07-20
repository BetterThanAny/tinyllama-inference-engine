from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from scripts import benchmark, benchmark_batch, source_snapshot, wsl_cuda_env

SOURCE_SHA256 = "a" * 64
FP16_SHA256 = "b" * 64


def sample() -> dict[str, float]:
    return {
        "prefill_compute_ms": 10.0,
        "prefill_transfer_ms": 1.0,
        "prefill_wall_ms": 12.0,
        "first_sampling_ms": 0.5,
        "ttft_ms": 12.5,
        "decode_compute_ms": 3.0,
        "decode_transfer_ms": 0.3,
        "decode_wall_ms": 4.0,
        "sampling_ms": 0.4,
        "tpot_ms": 2.2,
        "decode_compute_tokens_per_second": 1000.0,
        "decode_tokens_per_second": 700.0,
    }


def context_result(context: int = 128, samples: int = 2) -> dict[str, Any]:
    return {
        "gpu": benchmark.EXPECTED_GPU,
        "vram_bytes": 16 * 1024**3,
        "model_device_bytes": 3 * 1024**3,
        "engine_peak_device_bytes": 4 * 1024**3,
        "compute_capability": "8.6",
        "driver_version": 12080,
        "cuda_runtime_version": 12080,
        "model_source_sha256": SOURCE_SHA256,
        "model_fp16_sha256": FP16_SHA256,
        "dtype": "float16",
        "prompt_seed_text": benchmark.PROMPT_SEED_TEXT,
        "prompt_seed_token_ids": [1, 2, 3],
        "context_construction": benchmark.CONTEXT_CONSTRUCTION,
        "sampling": "greedy",
        "batch": 1,
        "context": context,
        "output_tokens": 32,
        "warmup": 3,
        "sample_count": samples,
        "rope_extrapolated_beyond_trained_context": context + 32 > 2048,
        "samples": [sample() for _ in range(samples)],
    }


def kernel_results() -> dict[str, Any]:
    names_and_shapes: list[tuple[str, dict[str, int]]] = [
        ("rms_norm", {}),
        ("rope", {}),
        ("softmax", {}),
        ("kv_update", {}),
        *(("attention_decode", {"context": context}) for context in (128, 512, 2048, 4096)),
        ("silu_multiply", {}),
        ("residual_add", {}),
        ("gemv", {}),
        ("gemm", {}),
        ("int8_weight_only_gemv", {}),
        ("int8_embedding_row", {}),
    ]
    return {
        "gpu": benchmark.EXPECTED_GPU,
        "timing": "cuda_event",
        "benchmarks": [
            {
                "name": name,
                "shape": shape,
                "warmup": benchmark.KERNEL_WARMUP,
                "samples": benchmark.KERNEL_SAMPLES,
                "median_ms": 0.1,
                "p95_ms": 0.2,
            }
            for name, shape in names_and_shapes
        ],
    }


class BenchmarkReportTests(unittest.TestCase):
    def test_validates_m4_batch_exit_threshold_and_metadata(self) -> None:
        result: dict[str, Any] = {
            "schema_version": 1,
            "mode": "benchmark",
            "dtype": "float16",
            "sampling": "greedy",
            "cuda_runtime_version": 12080,
            "model_fp16_sha256": FP16_SHA256,
            "context": 128,
            "output_tokens": 32,
            "warmup": 3,
            "samples": 10,
            "batch_1_total_tokens_per_second": 20.0,
            "batch_4_total_tokens_per_second": 40.0,
            "batch_4_over_batch_1": 2.0,
            "tokens_match": True,
            "kv_slots": 4,
            "passed": True,
        }
        benchmark_batch.validate_batch_result(result, FP16_SHA256)
        result["batch_4_over_batch_1"] = 1.49
        with self.assertRaisesRegex(ValueError, "1.5x"):
            benchmark_batch.validate_batch_result(result, FP16_SHA256)

    def test_nvidia_smi_uses_standard_wsl_mapping_when_missing_from_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wsl_lib = Path(directory)
            executable = wsl_lib / "nvidia-smi"
            executable.write_bytes(b"binary")
            with (
                mock.patch.object(benchmark.shutil, "which", return_value=None),
                mock.patch.object(benchmark.cuda_env, "is_wsl", return_value=True),
                mock.patch.object(benchmark.cuda_env, "WSL_LIB_DIR", wsl_lib),
            ):
                self.assertEqual(benchmark.nvidia_smi_executable(), str(executable))

    def test_monitored_command_uses_cuda_environment_wrapper(self) -> None:
        environment = wsl_cuda_env.CudaToolEnvironment(
            {"LD_LIBRARY_PATH": "/wsl/driver:/usr/lib/wsl/lib"},
            ("/usr/bin/env", "LD_AUDIT=/tmp/wsl_cuda_ld_audit.so"),
            Path("/tmp/wsl_cuda_ld_audit.so"),
        )
        process = mock.Mock()
        process.poll.return_value = 0
        process.communicate.return_value = ('{"ok": true}\n', "")
        process.returncode = 0
        with (
            mock.patch.object(benchmark.subprocess, "Popen", return_value=process) as popen,
            mock.patch.object(
                benchmark,
                "capture_gpu_snapshot",
                return_value={
                    "timestamp_utc": "2026-07-16T00:00:00+00:00",
                    "metrics": {
                        "temperature.gpu": "60",
                        "clocks.sm": "1500",
                        "clocks.mem": "7000",
                        "memory.used": "4096",
                        "utilization.gpu": "90",
                        "pstate": "P0",
                        "clocks_event_reasons.sw_thermal_slowdown": "Not Active",
                    },
                },
            ),
        ):
            result, monitor = benchmark.run_json_monitored(
                ["kernel_benchmarks"], "kernel_microbenchmarks", environment
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(monitor["sample_count"], 1)
        popen.assert_called_once_with(
            [
                "/usr/bin/env",
                "LD_AUDIT=/tmp/wsl_cuda_ld_audit.so",
                "kernel_benchmarks",
            ],
            stdout=benchmark.subprocess.PIPE,
            stderr=benchmark.subprocess.PIPE,
            text=True,
            env=environment.process_env,
        )

    def test_reads_pinned_model_checksums_from_manifest(self) -> None:
        manifest = {
            "schema_version": 1,
            "files": {"model.safetensors": {"sha256": SOURCE_SHA256}},
            "cuda_converted_format": {"file_sha256": FP16_SHA256},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(benchmark.pinned_model_checksums(path), (SOURCE_SHA256, FP16_SHA256))

            manifest["cuda_converted_format"]["file_sha256"] = "not-a-checksum"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid model_fp16_sha256"):
                benchmark.pinned_model_checksums(path)

    def test_source_baseline_requires_unchanged_verified_tree(self) -> None:
        snapshot = {"tree_sha256": "a" * 64, "git": {"commit": None}}
        self.assertTrue(benchmark.reproducible_source_baseline(snapshot, snapshot.copy()))
        self.assertFalse(
            benchmark.reproducible_source_baseline(
                snapshot, {"tree_sha256": "b" * 64, "git": {"commit": None}}
            )
        )

    def test_source_snapshot_detects_changed_and_unexpected_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            source = root / "src" / "example.cpp"
            source.write_text("int main() { return 0; }\n", encoding="utf-8")
            document = source_snapshot.create_snapshot(root)
            path = root / source_snapshot.SNAPSHOT_NAME
            path.write_text(json.dumps(document), encoding="utf-8")
            verified = source_snapshot.verify_snapshot(path, root)
            self.assertEqual(verified["tree_sha256"], document["tree_sha256"])

            source.write_text("int main() { return 1; }\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed=.*example.cpp"):
                source_snapshot.verify_snapshot(path, root)

            source.write_text("int main() { return 0; }\n", encoding="utf-8")
            (root / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected=.*unexpected.txt"):
                source_snapshot.verify_snapshot(path, root)

    def test_gpu_monitor_summarizes_in_workload_clock_and_temperature_range(self) -> None:
        observations = [
            {
                "timestamp_utc": f"2026-07-16T00:00:0{index}+00:00",
                "metrics": {
                    "temperature.gpu": temperature,
                    "clocks.sm": clock,
                    "clocks.mem": "7000",
                    "memory.used": memory,
                    "utilization.gpu": utilization,
                    "pstate": pstate,
                    "clocks_event_reasons.sw_thermal_slowdown": slowdown,
                },
            }
            for index, (temperature, clock, memory, utilization, pstate, slowdown) in enumerate(
                [
                    ("60", "1500", "4000", "95", "P0", "Not Active"),
                    ("78", "1200", "5000", "100", "P2", "Active"),
                ]
            )
        ]
        summary = benchmark.summarize_gpu_monitor("context_4096", observations)
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["temperature_gpu_c_max"], 78.0)
        self.assertEqual(summary["clocks_sm_mhz_min"], 1200.0)
        self.assertEqual(summary["pstates"], ["P0", "P2"])
        self.assertEqual(summary["software_thermal_slowdown_states"], ["Active", "Not Active"])

    def test_accepts_complete_context_and_computes_latency_summary(self) -> None:
        result = context_result()
        benchmark.validate_context_result(
            result,
            context=128,
            output_tokens=32,
            warmup=3,
            samples=2,
            expected_gpu=benchmark.EXPECTED_GPU,
            expected_source_sha256=SOURCE_SHA256,
            expected_fp16_sha256=FP16_SHA256,
        )
        summary = benchmark.summarize(result)
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["ttft_ms_median"], 12.5)
        self.assertEqual(summary["tpot_ms_p95"], 2.2)

    def test_accepts_4096_rope_extrapolation_boundary(self) -> None:
        result = context_result(context=4096)
        benchmark.validate_context_result(
            result,
            context=4096,
            output_tokens=32,
            warmup=3,
            samples=2,
            expected_gpu=benchmark.EXPECTED_GPU,
            expected_source_sha256=SOURCE_SHA256,
            expected_fp16_sha256=FP16_SHA256,
        )

    def test_rejects_sample_count_mismatch(self) -> None:
        result = context_result()
        result["samples"] = [sample()]
        with self.assertRaisesRegex(ValueError, "returned 1 samples, expected 2"):
            benchmark.validate_context_result(
                result,
                context=128,
                output_tokens=32,
                warmup=3,
                samples=2,
                expected_gpu=benchmark.EXPECTED_GPU,
                expected_source_sha256=SOURCE_SHA256,
                expected_fp16_sha256=FP16_SHA256,
            )

    def test_rejects_unreproducible_prompt_metadata(self) -> None:
        result = context_result()
        result["prompt_seed_token_ids"] = [1]
        with self.assertRaisesRegex(ValueError, "invalid prompt seed token IDs"):
            benchmark.validate_context_result(
                result,
                context=128,
                output_tokens=32,
                warmup=3,
                samples=2,
                expected_gpu=benchmark.EXPECTED_GPU,
                expected_source_sha256=SOURCE_SHA256,
                expected_fp16_sha256=FP16_SHA256,
            )

    def test_rejects_model_checksum_drift(self) -> None:
        result = context_result()
        result["model_fp16_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "model_fp16_sha256 mismatch"):
            benchmark.validate_context_result(
                result,
                context=128,
                output_tokens=32,
                warmup=3,
                samples=2,
                expected_gpu=benchmark.EXPECTED_GPU,
                expected_source_sha256=SOURCE_SHA256,
                expected_fp16_sha256=FP16_SHA256,
            )

    def test_rejects_non_finite_measurement(self) -> None:
        result = context_result()
        result["samples"][0]["ttft_ms"] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN, Inf, or negative"):
            benchmark.summarize(result)

    def test_accepts_complete_kernel_matrix_and_rejects_missing_family(self) -> None:
        result = kernel_results()
        benchmark.validate_kernel_results(result, benchmark.EXPECTED_GPU)
        incomplete = copy.deepcopy(result)
        incomplete["benchmarks"] = [
            item for item in incomplete["benchmarks"] if item["name"] != "gemm"
        ]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            benchmark.validate_kernel_results(incomplete, benchmark.EXPECTED_GPU)

    def test_markdown_reports_latency_tail_instead_of_favorable_throughput_tail(self) -> None:
        result = context_result()
        summary = benchmark.summarize(result)
        report = {
            "timestamp_utc": "2026-07-16T00:00:00+00:00",
            "source": {
                "before": {
                    "tree_sha256": "c" * 64,
                    "git": {"commit": None, "working_tree_dirty": True},
                }
            },
            "gpu_before": {
                "name": benchmark.EXPECTED_GPU,
                "memory.total": "16384",
                "driver_version": "1",
            },
            "cuda_toolkit_version": "Cuda compilation tools, release 12.8",
            "cuda_runtime_version": 12080,
            "power_mode": "test",
            "model_fp16_sha256": "b" * 64,
            "prompt_seed_text": benchmark.PROMPT_SEED_TEXT,
            "context_construction": benchmark.CONTEXT_CONSTRUCTION,
            "sampling": "greedy",
            "warmup": 3,
            "samples": 2,
            "acceptance": {"passed": True, "reason": "test"},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.md"
            benchmark.write_markdown(output, report, [summary])
            markdown = output.read_text(encoding="utf-8")
        self.assertIn("TTFT median ms", markdown)
        self.assertIn("Decode wall p95 ms", markdown)
        self.assertIn("MANUAL REVIEW REQUIRED", markdown)
        self.assertNotIn("p95 tok/s", markdown)


if __name__ == "__main__":
    unittest.main()
