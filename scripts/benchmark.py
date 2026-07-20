from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

try:
    import wsl_cuda_env as cuda_env
    from source_snapshot import verify_snapshot
except ModuleNotFoundError:
    from scripts import wsl_cuda_env as cuda_env
    from scripts.source_snapshot import verify_snapshot

EXPECTED_GPU = "NVIDIA GeForce RTX 3080 Laptop GPU"
KERNEL_WARMUP = 10
KERNEL_SAMPLES = 50
PROMPT_SEED_TEXT = "The capital of France is"
CONTEXT_CONSTRUCTION = "first seed token once, then cyclic repetition of remaining seed token IDs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the M2 FP16 CUDA baseline report")
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 512, 2048, 4096])
    parser.add_argument("--batch", type=int, default=1, choices=[1])
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--power-mode", required=True, help="Observed laptop AC/performance mode")
    parser.add_argument("--engine", type=Path, default=Path("build/cuda-release/tinyllama_cuda"))
    parser.add_argument(
        "--kernel-benchmarks",
        type=Path,
        default=Path("build/cuda-release/benchmarks/kernel_benchmarks"),
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/tinyllama-chat-v1.0"))
    parser.add_argument("--manifest", type=Path, default=Path("config/model_manifest.json"))
    parser.add_argument("--source-snapshot", type=Path, default=Path(".tlie-source-snapshot.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-gpu-name", default=EXPECTED_GPU)
    return parser.parse_args()


def parse_json_output(
    command: list[str], returncode: int, stdout: str, stderr: str
) -> dict[str, Any]:
    if returncode != 0:
        failure = f"command failed ({returncode}): {' '.join(command)}"
        raise RuntimeError(f"{failure}\n{stderr.strip()}")
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"command returned no JSON: {' '.join(command)}")
    return cast(dict[str, Any], json.loads(lines[-1]))


def nvidia_smi_executable() -> str:
    executable = shutil.which("nvidia-smi")
    if executable is not None:
        return executable
    wsl_executable = cuda_env.WSL_LIB_DIR / "nvidia-smi"
    if cuda_env.is_wsl() and wsl_executable.is_file():
        return str(wsl_executable)
    raise FileNotFoundError("nvidia-smi is required for GPU metadata and workload monitoring")


def gpu_snapshot() -> dict[str, str]:
    fields = [
        "name",
        "memory.total",
        "memory.used",
        "driver_version",
        "pstate",
        "power.limit",
        "temperature.gpu",
        "clocks.sm",
        "clocks.mem",
        "clocks_event_reasons.sw_thermal_slowdown",
        "utilization.gpu",
    ]
    completed = subprocess.run(
        [
            nvidia_smi_executable(),
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi metadata query failed: {completed.stderr.strip()}")
    rows = [row for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"M2 requires exactly one GPU, found {len(rows)}")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != len(fields):
        raise RuntimeError("nvidia-smi returned an unexpected metadata shape")
    return dict(zip(fields, values, strict=True))


def capture_gpu_snapshot(phase: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "metrics": gpu_snapshot(),
    }


def summarize_gpu_monitor(phase: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
    if not observations:
        raise ValueError(f"GPU monitor captured 0 samples for {phase}")

    def values(field: str) -> list[float]:
        result = [
            float(cast(dict[str, str], observation["metrics"])[field])
            for observation in observations
        ]
        if not all(math.isfinite(value) and value >= 0.0 for value in result):
            raise ValueError(f"GPU monitor field {field} is invalid for {phase}")
        return result

    temperatures = values("temperature.gpu")
    sm_clocks = values("clocks.sm")
    memory_clocks = values("clocks.mem")
    memory_used = values("memory.used")
    utilization = values("utilization.gpu")
    pstates = sorted(
        {cast(dict[str, str], observation["metrics"])["pstate"] for observation in observations}
    )
    thermal_slowdown_states = sorted(
        {
            cast(dict[str, str], observation["metrics"])["clocks_event_reasons.sw_thermal_slowdown"]
            for observation in observations
        }
    )
    if not set(thermal_slowdown_states) <= {"Active", "Not Active"}:
        raise ValueError(f"GPU thermal slowdown state is invalid for {phase}")
    return {
        "phase": phase,
        "poll_interval_seconds": 1.0,
        "sample_count": len(observations),
        "first_timestamp_utc": observations[0]["timestamp_utc"],
        "last_timestamp_utc": observations[-1]["timestamp_utc"],
        "temperature_gpu_c_min": min(temperatures),
        "temperature_gpu_c_max": max(temperatures),
        "clocks_sm_mhz_min": min(sm_clocks),
        "clocks_sm_mhz_max": max(sm_clocks),
        "clocks_memory_mhz_min": min(memory_clocks),
        "clocks_memory_mhz_max": max(memory_clocks),
        "memory_used_mib_max": max(memory_used),
        "utilization_gpu_percent_max": max(utilization),
        "pstates": pstates,
        "software_thermal_slowdown_states": thermal_slowdown_states,
    }


def run_json_monitored(
    command: list[str],
    phase: str,
    environment: cuda_env.CudaToolEnvironment | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    launched_command = environment.wrap_target(command) if environment is not None else command
    process_env = environment.process_env if environment is not None else None
    process = subprocess.Popen(
        launched_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=process_env,
    )
    observations: list[dict[str, Any]] = []
    try:
        observations.append(capture_gpu_snapshot(phase))
        while process.poll() is None:
            time.sleep(1.0)
            if process.poll() is None:
                observations.append(capture_gpu_snapshot(phase))
        stdout, stderr = process.communicate()
    except BaseException:
        process.kill()
        process.wait()
        raise
    returncode = process.returncode
    if returncode is None:
        raise RuntimeError(f"monitored command did not terminate: {' '.join(command)}")
    result = parse_json_output(launched_command, returncode, stdout, stderr)
    return result, summarize_gpu_monitor(phase, observations)


def cuda_toolkit_version() -> str:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise FileNotFoundError(
            "nvcc is required to record the CUDA toolkit version; "
            "do not install it globally without approval"
        )
    completed = subprocess.run([nvcc, "--version"], check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"nvcc version query failed: {completed.stderr.strip()}")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("nvcc version query returned no version text")
    return lines[-1]


def reproducible_source_baseline(before: dict[str, Any], after: dict[str, Any]) -> bool:
    tree = before.get("tree_sha256")
    return isinstance(tree, str) and len(tree) == 64 and before == after


def pinned_model_checksums(manifest_path: Path) -> tuple[str, str]:
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported model manifest schema")
    try:
        source_sha256 = cast(str, manifest["files"]["model.safetensors"]["sha256"])
        fp16_sha256 = cast(str, manifest["cuda_converted_format"]["file_sha256"])
    except (KeyError, TypeError) as error:
        raise ValueError("model manifest is missing pinned CUDA checksums") from error
    hexadecimal = set("0123456789abcdef")
    for name, value in (
        ("source_model_sha256", source_sha256),
        ("model_fp16_sha256", fp16_sha256),
    ):
        if not isinstance(value, str) or len(value) != 64 or not set(value) <= hexadecimal:
            raise ValueError(f"model manifest has an invalid {name}")
    return source_sha256, fp16_sha256


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def validate_kernel_results(result: dict[str, Any], expected_gpu: str) -> None:
    if result.get("gpu") != expected_gpu or result.get("timing") != "cuda_event":
        raise ValueError("kernel benchmark GPU or timing backend does not match M2")
    benchmarks = cast(list[dict[str, Any]], result.get("benchmarks"))
    if not isinstance(benchmarks, list):
        raise ValueError("kernel benchmark result has no benchmark list")
    expected_counts = {
        "rms_norm": 1,
        "rope": 1,
        "softmax": 1,
        "kv_update": 1,
        "attention_decode": 4,
        "silu_multiply": 1,
        "residual_add": 1,
        "gemv": 1,
        "gemm": 1,
        "int8_weight_only_gemv": 1,
        "int8_embedding_row": 1,
    }
    observed_counts = {name: 0 for name in expected_counts}
    for benchmark in benchmarks:
        name = str(benchmark.get("name"))
        if name not in observed_counts:
            raise ValueError(f"unexpected kernel benchmark family: {name}")
        observed_counts[name] += 1
        if benchmark.get("warmup") != KERNEL_WARMUP or benchmark.get("samples") != KERNEL_SAMPLES:
            raise ValueError(f"kernel benchmark {name} used an unexpected sample count")
        for field in ("median_ms", "p95_ms"):
            value = float(benchmark[field])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"kernel benchmark {name} field {field} is invalid")
    if observed_counts != expected_counts:
        raise ValueError(
            f"kernel benchmark families are incomplete: expected {expected_counts}, "
            f"found {observed_counts}"
        )
    attention_contexts = {
        int(cast(dict[str, Any], benchmark["shape"])["context"])
        for benchmark in benchmarks
        if benchmark["name"] == "attention_decode"
    }
    if attention_contexts != {128, 512, 2048, 4096}:
        raise ValueError("attention microbenchmarks do not cover all M2 contexts")


def validate_context_result(
    result: dict[str, Any],
    *,
    context: int,
    output_tokens: int,
    warmup: int,
    samples: int,
    expected_gpu: str,
    expected_source_sha256: str,
    expected_fp16_sha256: str,
) -> None:
    expected = {
        "gpu": expected_gpu,
        "dtype": "float16",
        "prompt_seed_text": PROMPT_SEED_TEXT,
        "context_construction": CONTEXT_CONSTRUCTION,
        "sampling": "greedy",
        "batch": 1,
        "context": context,
        "output_tokens": output_tokens,
        "warmup": warmup,
        "sample_count": samples,
        "rope_extrapolated_beyond_trained_context": context + output_tokens > 2048,
    }
    for field, expected_value in expected.items():
        if result.get(field) != expected_value:
            raise ValueError(
                f"context {context} field {field} mismatch: "
                f"expected {expected_value!r}, found {result.get(field)!r}"
            )
    observations = result.get("samples")
    if not isinstance(observations, list) or len(observations) != samples:
        observed_count = len(observations) if isinstance(observations, list) else 0
        raise ValueError(f"context {context} returned {observed_count} samples, expected {samples}")
    seed_token_ids = result.get("prompt_seed_token_ids")
    if (
        not isinstance(seed_token_ids, list)
        or len(seed_token_ids) < 2
        or not all(isinstance(token, int) and token >= 0 for token in seed_token_ids)
    ):
        raise ValueError(f"context {context} has invalid prompt seed token IDs")
    expected_checksums = {
        "model_source_sha256": expected_source_sha256,
        "model_fp16_sha256": expected_fp16_sha256,
    }
    for field, expected_checksum in expected_checksums.items():
        if result.get(field) != expected_checksum:
            raise ValueError(
                f"context {context} {field} mismatch: expected {expected_checksum}, "
                f"found {result.get(field)}"
            )
    if result.get("compute_capability") != "8.6":
        raise ValueError(f"context {context} did not run on the required SM 86 GPU")
    if int(result.get("cuda_runtime_version", 0)) <= 0:
        raise ValueError(f"context {context} has no CUDA runtime version")
    vram_bytes = int(result.get("vram_bytes", 0))
    model_bytes = int(result.get("model_device_bytes", 0))
    peak_bytes = int(result.get("engine_peak_device_bytes", 0))
    if not (0 < model_bytes <= peak_bytes <= vram_bytes):
        raise ValueError(f"context {context} has inconsistent CUDA memory accounting")


def validate_consistent_runtime(results: list[dict[str, Any]]) -> None:
    if not results:
        raise ValueError("benchmark returned 0 context results")
    identity_fields = [
        "gpu",
        "vram_bytes",
        "compute_capability",
        "driver_version",
        "cuda_runtime_version",
        "model_source_sha256",
        "model_fp16_sha256",
        "dtype",
        "prompt_seed_text",
        "prompt_seed_token_ids",
        "context_construction",
        "sampling",
        "batch",
    ]
    baseline = results[0]
    for result in results[1:]:
        for field in identity_fields:
            if result.get(field) != baseline.get(field):
                raise ValueError(f"benchmark runtime metadata changed between contexts: {field}")


def summarize(context_result: dict[str, Any]) -> dict[str, Any]:
    samples = cast(list[dict[str, Any]], context_result["samples"])
    if not samples:
        raise ValueError("benchmark returned 0 samples")
    fields = [
        "prefill_compute_ms",
        "prefill_transfer_ms",
        "prefill_wall_ms",
        "first_sampling_ms",
        "ttft_ms",
        "decode_compute_ms",
        "decode_transfer_ms",
        "decode_wall_ms",
        "sampling_ms",
        "tpot_ms",
        "decode_compute_tokens_per_second",
        "decode_tokens_per_second",
    ]
    summary: dict[str, Any] = {
        "context": context_result["context"],
        "batch": context_result["batch"],
        "sample_count": len(samples),
        "model_device_bytes": context_result["model_device_bytes"],
        "engine_peak_device_bytes": context_result["engine_peak_device_bytes"],
        "rope_extrapolated_beyond_trained_context": context_result[
            "rope_extrapolated_beyond_trained_context"
        ],
    }
    for field in fields:
        values = [float(sample[field]) for sample in samples]
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError(f"benchmark field {field} contains NaN, Inf, or negative values")
        summary[f"{field}_median"] = statistics.median(values)
        summary[f"{field}_p95"] = percentile(values, 0.95)
    return summary


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def write_markdown(path: Path, report: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    before = cast(dict[str, str], report["gpu_before"])
    source_before = cast(dict[str, Any], report["source"]["before"])
    git_before = cast(dict[str, Any], source_before["git"])
    peak_gib = max(float(summary["engine_peak_device_bytes"]) for summary in summaries) / 1024**3
    toolkit = report["cuda_toolkit_version"]
    runtime = report["cuda_runtime_version"]
    lines = [
        "# M2 FP16 CUDA baseline",
        "",
        f"- Timestamp: `{report['timestamp_utc']}`",
        f"- Git commit: `{git_before['commit']}`; dirty: `{git_before['working_tree_dirty']}`",
        f"- Mac source tree SHA-256: `{source_before['tree_sha256']}`",
        f"- GPU: `{before['name']}`; VRAM: `{before['memory.total']} MiB`",
        f"- Driver: `{before['driver_version']}`; CUDA toolkit: `{toolkit}`; runtime: `{runtime}`",
        f"- Power mode: `{report['power_mode']}`",
        f"- Model FP16 SHA-256: `{report['model_fp16_sha256']}`",
        f"- Prompt seed: `{report['prompt_seed_text']}`; sampling: `{report['sampling']}`",
        f"- Context construction: `{report['context_construction']}`",
        f"- Warmup/samples: `{report['warmup']}` / `{report['samples']}`",
        f"- Maximum observed engine allocation: `{peak_gib:.3f} GiB`",
        "",
        "| Context | RoPE extrapolated | TTFT median ms | TPOT median ms "
        "| Decode median tok/s | Decode wall p95 ms |",
        "|---:|:---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            (
                "| {context} | {extrapolated} | {ttft:.3f} | {tpot:.3f} "
                "| {tokens:.3f} | {p95:.3f} |"
            ).format(
                context=summary["context"],
                extrapolated="yes" if summary["rope_extrapolated_beyond_trained_context"] else "no",
                ttft=summary["ttft_ms_median"],
                tpot=summary["tpot_ms_median"],
                tokens=summary["decode_tokens_per_second_median"],
                p95=summary["decode_wall_ms_p95"],
            )
        )
    lines.extend(
        [
            "",
            "The 4096-token row is an explicit RoPE-extrapolated memory-safety/performance "
            "stress case, not a quality claim.",
            "",
            "Automated acceptance: **{}** — {}".format(
                "PASS" if report["acceptance"]["passed"] else "FAIL",
                report["acceptance"]["reason"],
            ),
            "",
            "Thermal/clock acceptance: **MANUAL REVIEW REQUIRED** — inspect the in-workload "
            "`gpu_workload_monitors` ranges before accepting this baseline.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if sorted(set(args.contexts)) != [128, 512, 2048, 4096]:
        raise ValueError("M2 report requires exactly contexts 128, 512, 2048, and 4096")
    if args.output_tokens < 2 or args.output_tokens > 256 or args.warmup <= 0 or args.samples <= 0:
        raise ValueError("output-tokens must be 2..256 and warmup/samples must be positive")
    for executable in (args.engine, args.kernel_benchmarks):
        if not executable.is_file():
            raise FileNotFoundError(f"required CUDA executable is missing: {executable}")
    cuda_runtime_dir = args.engine.parent / "benchmark-runtime"
    cuda_runtime_dir.mkdir(parents=True, exist_ok=True)
    cuda_environment = cuda_env.prepare_cuda_tool_environment(cuda_runtime_dir)
    expected_source_sha256, expected_fp16_sha256 = pinned_model_checksums(args.manifest)
    toolkit_version = cuda_toolkit_version()
    source_before = verify_snapshot(args.source_snapshot, Path.cwd())
    gpu_snapshots = [capture_gpu_snapshot("before")]
    before = cast(dict[str, str], gpu_snapshots[0]["metrics"])
    if before["name"] != args.require_gpu_name:
        raise RuntimeError(
            f"designated GPU mismatch: expected {args.require_gpu_name!r}, found {before['name']!r}"
        )
    kernel_results, kernel_monitor = run_json_monitored(
        [
            str(args.kernel_benchmarks),
            "--warmup",
            str(KERNEL_WARMUP),
            "--samples",
            str(KERNEL_SAMPLES),
        ],
        "kernel_microbenchmarks",
        cuda_environment,
    )
    validate_kernel_results(kernel_results, args.require_gpu_name)
    gpu_workload_monitors = [kernel_monitor]
    gpu_snapshots.append(capture_gpu_snapshot("after_kernel_microbenchmarks"))
    raw_contexts: list[dict[str, Any]] = []
    for context in sorted(args.contexts):
        result, context_monitor = run_json_monitored(
            [
                str(args.engine),
                "benchmark",
                str(args.model_dir),
                str(context),
                str(args.output_tokens),
                str(args.warmup),
                str(args.samples),
            ],
            f"context_{context}",
            cuda_environment,
        )
        validate_context_result(
            result,
            context=context,
            output_tokens=args.output_tokens,
            warmup=args.warmup,
            samples=args.samples,
            expected_gpu=args.require_gpu_name,
            expected_source_sha256=expected_source_sha256,
            expected_fp16_sha256=expected_fp16_sha256,
        )
        raw_contexts.append(result)
        gpu_workload_monitors.append(context_monitor)
        gpu_snapshots.append(capture_gpu_snapshot(f"after_context_{context}"))
    validate_consistent_runtime(raw_contexts)
    summaries = [summarize(result) for result in raw_contexts]
    source_after = verify_snapshot(args.source_snapshot, Path.cwd())
    after = cast(dict[str, str], gpu_snapshots[-1]["metrics"])
    baseline = next(summary for summary in summaries if summary["context"] == 128)
    throughput = float(baseline["decode_tokens_per_second_median"])
    throughput_passed = throughput >= 90.0
    source_passed = reproducible_source_baseline(source_before, source_after)
    acceptance_passed = throughput_passed and source_passed
    comparison = ">=" if throughput_passed else "<"
    reason = f"context=128 median FP16 decode {throughput:.3f} tok/s {comparison} 90 tok/s"
    reason += (
        "; verified Mac source snapshot remained unchanged"
        if source_passed
        else "; Mac source snapshot missing, invalid, or changed"
    )
    first = raw_contexts[0]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("benchmarks/results") / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 2,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "os": platform.platform(),
        "source": {
            "snapshot": str(args.source_snapshot),
            "before": source_before,
            "after": source_after,
        },
        "gpu_before": before,
        "gpu_after": after,
        "gpu_snapshots": gpu_snapshots,
        "gpu_workload_monitors": gpu_workload_monitors,
        "power_mode": args.power_mode,
        "cuda_toolkit_version": toolkit_version,
        "cuda_runtime_version": first["cuda_runtime_version"],
        "compute_capability": first["compute_capability"],
        "model_source_sha256": first["model_source_sha256"],
        "model_fp16_sha256": first["model_fp16_sha256"],
        "model_manifest": str(args.manifest),
        "dtype": "float16",
        "prompt_seed_text": first["prompt_seed_text"],
        "prompt_seed_token_ids": first["prompt_seed_token_ids"],
        "context_construction": first["context_construction"],
        "sampling": first["sampling"],
        "batch": 1,
        "output_tokens": args.output_tokens,
        "warmup": args.warmup,
        "samples": args.samples,
        "contexts": raw_contexts,
        "summary": summaries,
        "kernel_microbenchmarks": kernel_results,
        "timing_boundaries": {
            "model_load_included": False,
            "warmup_included": False,
            "prefill": "all fixed input-token Forward calls",
            "ttft": "prefill wall time plus first greedy sampling",
            "decode": "Forward calls and greedy sampling between first and last output token",
            "gpu_compute": "CUDA events",
            "service_wall_and_sampling": "host monotonic clock",
        },
        "acceptance": {
            "passed": acceptance_passed,
            "reason": reason,
            "throughput_passed": throughput_passed,
            "reproducible_source_baseline_passed": source_passed,
            "scope": "automated_preconditions_only",
            "thermal_clock_review_required": True,
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "summary.csv", summaries)
    write_markdown(output_dir / "report.md", report, summaries)
    print(json.dumps({"report_dir": str(output_dir), "acceptance": report["acceptance"]}))
    if not acceptance_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
