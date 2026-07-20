from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

try:
    import benchmark as m2_benchmark
    import wsl_cuda_env as cuda_env
    from source_snapshot import verify_snapshot
except ModuleNotFoundError:
    from scripts import benchmark as m2_benchmark
    from scripts import wsl_cuda_env as cuda_env
    from scripts.source_snapshot import verify_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FP16 and W8A16 on fixed M3 workloads")
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 4096])
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="run only context 128 for a thermally isolated M5 cross-engine comparison",
    )
    parser.add_argument(
        "--int8-first",
        action="store_true",
        help="run INT8 before FP16 in comparison-only mode for thermal isolation",
    )
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--power-mode", required=True)
    parser.add_argument("--engine", type=Path, default=Path("build/cuda-release/tinyllama_cuda"))
    parser.add_argument("--manifest", type=Path, default=Path("config/model_manifest.json"))
    parser.add_argument("--source-snapshot", type=Path, default=Path(".tlie-source-snapshot.json"))
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_checksums(path: Path) -> dict[str, str]:
    document = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    return {
        "source": cast(str, document["files"]["model.safetensors"]["sha256"]),
        "float16": cast(str, document["cuda_converted_format"]["file_sha256"]),
        "int8_weight_only": cast(str, document["int8_converted_format"]["file_sha256"]),
    }


def validate_contexts(contexts: list[int], comparison_only: bool) -> None:
    expected = [128] if comparison_only else [128, 4096]
    if sorted(set(contexts)) != expected:
        raise ValueError(
            "M5 comparison-only requires context 128; "
            "the default M3 comparison requires contexts 128 and 4096"
        )


def validate_result(
    result: dict[str, Any],
    mode: str,
    context: int,
    output_tokens: int,
    warmup: int,
    samples: int,
    checksums: dict[str, str],
) -> None:
    expected_dtype = "float16" if mode == "float16" else "int8_weight_only"
    checksum_field = "model_fp16_sha256" if mode == "float16" else "model_int8_sha256"
    expected = {
        "gpu": m2_benchmark.EXPECTED_GPU,
        "dtype": expected_dtype,
        "context": context,
        "output_tokens": output_tokens,
        "warmup": warmup,
        "sample_count": samples,
        "batch": 1,
        "model_source_sha256": checksums["source"],
        checksum_field: checksums[mode],
        "model_weight_sha256": checksums[mode],
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f"{mode} context {context} returned invalid {key}: {result.get(key)!r}"
            )
    measurements = cast(list[dict[str, Any]], result.get("samples"))
    if len(measurements) != samples:
        raise ValueError(f"{mode} context {context} returned an invalid sample count")
    if not all(
        math.isfinite(float(value)) and float(value) >= 0.0
        for sample in measurements
        for value in sample.values()
    ):
        raise ValueError(f"{mode} context {context} contains NaN, Inf, or negative measurements")
    before = result.get("device_allocation_count_before_workload")
    after = result.get("device_allocation_count_after_workload")
    if not isinstance(before, int) or before <= 0 or after != before:
        raise ValueError(f"{mode} context {context} changed device allocation count")
    if not isinstance(result.get("kv_cache_bytes"), int) or result["kv_cache_bytes"] <= 0:
        raise ValueError(f"{mode} context {context} returned invalid KV Cache bytes")


def acceptance(
    fp16_summary: dict[str, Any], int8_summary: dict[str, Any]
) -> dict[str, float | bool]:
    fp16_throughput = float(fp16_summary["decode_tokens_per_second_median"])
    int8_throughput = float(int8_summary["decode_tokens_per_second_median"])
    fp16_memory = float(fp16_summary["engine_peak_device_bytes"])
    int8_memory = float(int8_summary["engine_peak_device_bytes"])
    throughput_gain = int8_throughput / fp16_throughput - 1.0
    memory_reduction = 1.0 - int8_memory / fp16_memory
    return {
        "throughput_gain_fraction": throughput_gain,
        "memory_reduction_fraction": memory_reduction,
        "throughput_threshold_passed": throughput_gain >= 0.20,
        "memory_threshold_passed": memory_reduction >= 0.25,
        "passed": throughput_gain >= 0.20 or memory_reduction >= 0.25,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "mode",
        "context",
        "sample_count",
        "ttft_ms_median",
        "tpot_ms_median",
        "decode_tokens_per_second_median",
        "decode_wall_ms_p95",
        "model_device_bytes",
        "engine_peak_device_bytes",
        "kv_cache_bytes",
        "device_allocation_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# M3 FP16 vs INT8 weight-only report",
        "",
        "| Mode | Context | TTFT median ms | TPOT median ms | Decode median tok/s | Peak MiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in cast(list[dict[str, Any]], report["summary"]):
        lines.append(
            f"| {row['mode']} | {row['context']} | {row['ttft_ms_median']:.3f} | "
            f"{row['tpot_ms_median']:.3f} | {row['decode_tokens_per_second_median']:.3f} | "
            f"{row['engine_peak_device_bytes'] / 1024**2:.3f} |"
        )
    result = cast(dict[str, Any], report["acceptance"])
    lines.extend(
        [
            "",
            f"Throughput gain at context 128: {result['throughput_gain_fraction']:.3%}.",
            f"Peak-memory reduction at context 128: {result['memory_reduction_fraction']:.3%}.",
            "",
            f"M3 threshold: **{'PASS' if result['passed'] else 'NON-FINDING'}**.",
            "",
            "Thermal/clock ranges remain qualified by the per-workload monitor records.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_contexts(args.contexts, args.comparison_only)
    if args.int8_first and not args.comparison_only:
        raise ValueError("int8-first is only valid with comparison-only")
    if args.output_tokens < 2 or args.warmup <= 0 or args.samples <= 0:
        raise ValueError("output-tokens must be >=2 and warmup/samples must be positive")
    if not args.engine.is_file():
        raise FileNotFoundError(f"CUDA engine is missing: {args.engine}")
    runtime_dir = args.engine.parent / "m3-benchmark-runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    environment = cuda_env.prepare_cuda_tool_environment(runtime_dir)
    checksums = load_checksums(args.manifest)
    source_before = verify_snapshot(args.source_snapshot, Path.cwd())
    raw: list[dict[str, Any]] = []
    monitors: list[dict[str, Any]] = []
    modes = [("float16", "benchmark"), ("int8_weight_only", "benchmark-int8")]
    if args.int8_first:
        modes.reverse()
    for context in sorted(args.contexts):
        for mode, command in modes:
            result, monitor = m2_benchmark.run_json_monitored(
                [
                    str(args.engine),
                    command,
                    "models/tinyllama-chat-v1.0",
                    str(context),
                    str(args.output_tokens),
                    str(args.warmup),
                    str(args.samples),
                ],
                f"{mode}_context_{context}",
                environment,
            )
            validate_result(
                result, mode, context, args.output_tokens, args.warmup, args.samples, checksums
            )
            result["mode"] = mode
            raw.append(result)
            monitors.append(monitor)
    source_after = verify_snapshot(args.source_snapshot, Path.cwd())
    if source_before["tree_sha256"] != source_after["tree_sha256"]:
        raise ValueError("Mac source snapshot changed during M3 comparison")
    summaries: list[dict[str, Any]] = []
    for result in raw:
        summary = m2_benchmark.summarize(result)
        summary.update(
            {
                "mode": result["mode"],
                "model_device_bytes": result["model_device_bytes"],
                "engine_peak_device_bytes": result["engine_peak_device_bytes"],
                "kv_cache_bytes": result["kv_cache_bytes"],
                "device_allocation_count": result["device_allocation_count_after_workload"],
            }
        )
        summaries.append(summary)
    fp16_128 = next(row for row in summaries if row["mode"] == "float16" and row["context"] == 128)
    int8_128 = next(
        row for row in summaries if row["mode"] == "int8_weight_only" and row["context"] == 128
    )
    accepted = acceptance(fp16_128, int8_128)
    allocation_stable = all(
        result["device_allocation_count_before_workload"]
        == result["device_allocation_count_after_workload"]
        for result in raw
    )
    accepted["allocation_count_stable"] = allocation_stable
    accepted["passed"] = bool(accepted["passed"]) and allocation_stable
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("benchmarks/results") / f"{timestamp}-m3-int8"
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source": {"before": source_before, "after": source_after},
        "power_mode": args.power_mode,
        "cuda_toolkit_version": m2_benchmark.cuda_toolkit_version(),
        "checksums": checksums,
        "workload": {
            "contexts": sorted(args.contexts),
            "comparison_only": args.comparison_only,
            "int8_first": args.int8_first,
            "output_tokens": args.output_tokens,
            "warmup": args.warmup,
            "samples": args.samples,
            "prompt_seed_text": m2_benchmark.PROMPT_SEED_TEXT,
            "context_construction": m2_benchmark.CONTEXT_CONSTRUCTION,
            "sampling": "greedy",
        },
        "raw": raw,
        "summary": summaries,
        "gpu_workload_monitors": monitors,
        "acceptance": accepted,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "summary.csv", summaries)
    write_markdown(output_dir / "report.md", report)
    print(json.dumps({"report_dir": str(output_dir), "acceptance": accepted}))
    if not bool(accepted["passed"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
