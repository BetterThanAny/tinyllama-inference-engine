from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import benchmark
    import openai_smoke_test as smoke
    import wsl_cuda_env as cuda_env
    from source_snapshot import verify_snapshot
except ModuleNotFoundError:
    from scripts import benchmark
    from scripts import openai_smoke_test as smoke
    from scripts import wsl_cuda_env as cuda_env
    from scripts.source_snapshot import verify_snapshot


PROMPTS = {
    "short": "State one fact about France.",
    "medium": "Summarize this sequence briefly: " + "alpha beta gamma delta " * 24,
    "long": "Continue this bounded context: " + "one two three four five six seven eight " * 42,
}
FORMAL_CONCURRENCY = 20
FORMAL_DURATION_SECONDS = 1800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the M5 mixed-context stability load")
    parser.add_argument("--server", type=Path, default=Path("build/cuda-release/tinyllama_server"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/tinyllama-chat-v1.0"))
    parser.add_argument("--source-snapshot", type=Path, default=Path(".tlie-source-snapshot.json"))
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--duration", type=int, default=1800)
    parser.add_argument("--power-mode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_memory_stability(samples: list[float], limit_mib: float = 64.0) -> dict[str, Any]:
    if len(samples) < 8 or not all(math.isfinite(value) and value >= 0.0 for value in samples):
        raise ValueError("stability run has too few valid GPU memory samples")
    discarded_warmup = min(60, max(1, len(samples) // 10))
    settled = samples[discarded_warmup:]
    window = max(2, min(30, len(settled) // 4))
    first = statistics.median(settled[:window])
    last = statistics.median(settled[-window:])
    growth = last - first
    return {
        "sample_count": len(samples),
        "discarded_warmup_samples": discarded_warmup,
        "window_samples": window,
        "first_window_median_mib": first,
        "last_window_median_mib": last,
        "growth_mib": growth,
        "limit_mib": limit_mib,
        "passed": growth <= limit_mib,
    }


def finite_response_metrics(document: dict[str, Any]) -> bool:
    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        return False
    return all(
        math.isfinite(float(metrics.get(field, float("nan"))))
        and float(metrics.get(field, -1.0)) >= 0.0
        for field in ("queue_ms", "ttft_ms", "tpot_ms", "output_tokens_per_second")
    )


def validate_formal_workload(concurrency: int, duration: int) -> None:
    if concurrency != FORMAL_CONCURRENCY or duration != FORMAL_DURATION_SECONDS:
        raise ValueError(
            f"M5 formal stability requires concurrency {FORMAL_CONCURRENCY} and "
            f"duration {FORMAL_DURATION_SECONDS} seconds"
        )


def validate_request_accounting(metrics: dict[str, Any]) -> dict[str, Any]:
    fields = ("submitted_total", "completed_total", "cancelled_total", "failed_total")
    values = {field: int(metrics.get(field, -1)) for field in fields}
    nonnegative = all(value >= 0 for value in values.values())
    terminal_total = values["completed_total"] + values["cancelled_total"] + values["failed_total"]
    return {
        **values,
        "terminal_total": terminal_total,
        "passed": (
            nonnegative
            and values["failed_total"] == 0
            and terminal_total == values["submitted_total"]
        ),
    }


def worker(
    port: int, worker_id: int, deadline: float, counters: dict[str, int], lock: threading.Lock
) -> None:
    iteration = 0
    prompt_items = tuple(PROMPTS.items())
    while time.monotonic() < deadline:
        prompt_class, prompt = prompt_items[(worker_id + iteration) % len(prompt_items)]
        try:
            if iteration % 17 == 0:
                outcome = smoke.streaming_completion(port, prompt, cancel_after_first=True)
                if outcome != "cancelled":
                    raise RuntimeError(f"unexpected cancellation outcome: {outcome}")
                key = "cancelled"
            elif iteration % 19 == 0:
                status, document = smoke.request_json(
                    "POST",
                    port,
                    "/v1/chat/completions",
                    smoke.completion_body(prompt, max_tokens=8, timeout_ms=1),
                )
                if status != 408 or document.get("error", {}).get("code") != "request_cancelled":
                    raise RuntimeError(f"timeout path returned HTTP {status}: {document}")
                key = "timeout"
            else:
                status, document = smoke.request_json(
                    "POST",
                    port,
                    "/v1/chat/completions",
                    smoke.completion_body(prompt, max_tokens=4, timeout_ms=120_000),
                    timeout=150.0,
                )
                if status != 200 or not finite_response_metrics(document):
                    raise RuntimeError(f"completion returned HTTP {status}: {document}")
                key = f"completed_{prompt_class}"
            with lock:
                counters[key] = counters.get(key, 0) + 1
        except Exception:
            with lock:
                counters["unexpected_failures"] = counters.get("unexpected_failures", 0) + 1
            raise
        iteration += 1


def main() -> None:
    args = parse_args()
    validate_formal_workload(args.concurrency, args.duration)
    if not args.server.is_file():
        raise FileNotFoundError(f"server executable is missing: {args.server}")
    source_before = verify_snapshot(args.source_snapshot, Path.cwd())
    environment_directory = args.server.parent / "m5-load-runtime"
    environment_directory.mkdir(parents=True, exist_ok=True)
    environment = cuda_env.prepare_cuda_tool_environment(environment_directory)
    started = time.monotonic()
    counters: dict[str, int] = {}
    lock = threading.Lock()
    gpu_samples: list[dict[str, Any]] = []
    scheduler_samples: list[dict[str, Any]] = []
    metrics_poll_failures = 0
    consecutive_metrics_poll_failures = 0
    maximum_consecutive_metrics_poll_failures = 0
    with subprocess.Popen(
        environment.wrap_target(
            [str(args.server), "--port", str(args.port), "--model-dir", str(args.model_dir)]
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment.process_env,
    ) as server:
        try:
            smoke.wait_ready(server, args.port)
            deadline = time.monotonic() + args.duration
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [
                    executor.submit(worker, args.port, worker_id, deadline, counters, lock)
                    for worker_id in range(args.concurrency)
                ]
                while any(not future.done() for future in futures):
                    if server.poll() is not None:
                        stdout, _ = server.communicate()
                        raise RuntimeError(f"server crashed during stability load: {stdout}")
                    gpu_samples.append(benchmark.capture_gpu_snapshot("m5_stability"))
                    try:
                        status, metrics = smoke.request_json(
                            "GET", args.port, "/metrics", timeout=5.0
                        )
                        if status != 200:
                            raise RuntimeError(f"metrics endpoint returned HTTP {status}")
                    except (OSError, TimeoutError):
                        metrics_poll_failures += 1
                        consecutive_metrics_poll_failures += 1
                        maximum_consecutive_metrics_poll_failures = max(
                            maximum_consecutive_metrics_poll_failures,
                            consecutive_metrics_poll_failures,
                        )
                    else:
                        consecutive_metrics_poll_failures = 0
                        metrics["timestamp_monotonic"] = time.monotonic()
                        scheduler_samples.append(metrics)
                    time.sleep(1.0)
                for future in futures:
                    future.result()
            final_metrics = smoke.wait_drained(args.port)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10.0)
    elapsed = time.monotonic() - started
    memory_samples = [float(sample["metrics"]["memory.used"]) for sample in gpu_samples]
    memory = validate_memory_stability(memory_samples)
    source_after = verify_snapshot(args.source_snapshot, Path.cwd())
    no_leaks = all(
        int(final_metrics[field]) == 0
        for field in ("queued_requests", "active_sequences", "kv_blocks_used")
    )
    request_accounting = validate_request_accounting(final_metrics)
    completed = sum(value for key, value in counters.items() if key.startswith("completed_"))
    all_prompt_classes_completed = all(counters.get(f"completed_{name}", 0) > 0 for name in PROMPTS)
    passed = (
        elapsed >= args.duration
        and completed > 0
        and all_prompt_classes_completed
        and counters.get("unexpected_failures", 0) == 0
        and len(scheduler_samples) >= 10
        and bool(memory["passed"])
        and no_leaks
        and bool(request_accounting["passed"])
        and source_before == source_after
    )
    report = {
        "schema_version": 1,
        "milestone": "M5",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "duration_requested_seconds": args.duration,
        "duration_observed_seconds": elapsed,
        "concurrency": args.concurrency,
        "power_mode": args.power_mode,
        "prompt_classes": PROMPTS,
        "outcomes": counters,
        "gpu_before": gpu_samples[0]["metrics"],
        "gpu_after": gpu_samples[-1]["metrics"],
        "gpu_monitor": benchmark.summarize_gpu_monitor("m5_stability", gpu_samples),
        "memory_stability": memory,
        "scheduler_sample_count": len(scheduler_samples),
        "metrics_poll_failures": metrics_poll_failures,
        "maximum_consecutive_metrics_poll_failures": (maximum_consecutive_metrics_poll_failures),
        "scheduler_peaks": {
            field: max(int(sample[field]) for sample in scheduler_samples)
            for field in ("queued_requests", "active_sequences", "kv_blocks_used")
        },
        "final_metrics": final_metrics,
        "request_accounting": request_accounting,
        "source": {"before": source_before, "after": source_after},
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": passed, "outcomes": counters}))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
