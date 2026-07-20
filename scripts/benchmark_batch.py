from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

try:
    import benchmark
    import wsl_cuda_env as cuda_env
    from source_snapshot import verify_snapshot
except ModuleNotFoundError:
    from scripts import benchmark
    from scripts import wsl_cuda_env as cuda_env
    from scripts.source_snapshot import verify_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the M4 Batch 1/4 CUDA report")
    parser.add_argument(
        "--target", type=Path, default=Path("build/cuda-release/benchmarks/batch_benchmark")
    )
    parser.add_argument("--manifest", type=Path, default=Path("config/model_manifest.json"))
    parser.add_argument("--source-snapshot", type=Path, default=Path(".tlie-source-snapshot.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--power-mode", required=True)
    parser.add_argument("--require-gpu-name", default=benchmark.EXPECTED_GPU)
    return parser.parse_args()


def validate_batch_result(result: dict[str, Any], expected_fp16_sha256: str) -> None:
    expected = {
        "schema_version": 1,
        "mode": "benchmark",
        "dtype": "float16",
        "sampling": "greedy",
        "context": 128,
        "output_tokens": 32,
        "warmup": 3,
        "samples": 10,
        "kv_slots": 4,
        "model_fp16_sha256": expected_fp16_sha256,
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise ValueError(
                f"batch benchmark field {field} mismatch: expected {value!r}, "
                f"found {result.get(field)!r}"
            )
    numeric_fields = (
        "batch_1_total_tokens_per_second",
        "batch_4_total_tokens_per_second",
        "batch_4_over_batch_1",
    )
    for field in numeric_fields:
        value = float(result.get(field, 0.0))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"batch benchmark field {field} is invalid")
    if int(result.get("cuda_runtime_version", 0)) <= 0:
        raise ValueError("batch benchmark did not record the CUDA runtime version")
    if result.get("tokens_match") is not True:
        raise ValueError("Batch 4 generated tokens differ from Batch 1")
    if float(result["batch_4_over_batch_1"]) < 1.5 or result.get("passed") is not True:
        raise ValueError("Batch 4 throughput did not meet the 1.5x exit threshold")


def main() -> None:
    args = parse_args()
    if not args.target.is_file():
        raise FileNotFoundError(f"required CUDA executable is missing: {args.target}")
    _, expected_fp16_sha256 = benchmark.pinned_model_checksums(args.manifest)
    source_before = verify_snapshot(args.source_snapshot, Path.cwd())
    gpu_before_snapshot = benchmark.capture_gpu_snapshot("before_batch_benchmark")
    gpu_before = cast(dict[str, str], gpu_before_snapshot["metrics"])
    if gpu_before["name"] != args.require_gpu_name:
        raise RuntimeError(
            f"designated GPU mismatch: expected {args.require_gpu_name!r}, "
            f"found {gpu_before['name']!r}"
        )
    runtime_dir = args.target.parent / "batch-benchmark-runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    environment = cuda_env.prepare_cuda_tool_environment(runtime_dir)
    result, monitor = benchmark.run_json_monitored(
        [str(args.target), "--benchmark"], "batch_1_and_4", environment
    )
    validate_batch_result(result, expected_fp16_sha256)
    gpu_after_snapshot = benchmark.capture_gpu_snapshot("after_batch_benchmark")
    source_after = verify_snapshot(args.source_snapshot, Path.cwd())
    source_reproducible = benchmark.reproducible_source_baseline(source_before, source_after)
    if not source_reproducible:
        raise RuntimeError("source tree changed during the batch benchmark")
    report = {
        "schema_version": 1,
        "milestone": "M4",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "power_mode": args.power_mode,
        "cuda_toolkit_version": benchmark.cuda_toolkit_version(),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after_snapshot["metrics"],
        "gpu_workload_monitor": monitor,
        "source": {"before": source_before, "after": source_after},
        "source_reproducible": source_reproducible,
        "benchmark": result,
        "acceptance": {
            "threshold": "Batch 4 total throughput >= 1.5x Batch 1",
            "passed": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "acceptance": report["acceptance"]}))


if __name__ == "__main__":
    main()
