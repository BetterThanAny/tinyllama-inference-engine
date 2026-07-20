from __future__ import annotations

import argparse
import http.client
import json
import math
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

try:
    import benchmark
    import wsl_cuda_env as cuda_env
    from model_assets import sha256_file
    from source_snapshot import verify_snapshot
except ModuleNotFoundError:
    from scripts import benchmark
    from scripts import wsl_cuda_env as cuda_env
    from scripts.model_assets import sha256_file
    from scripts.source_snapshot import verify_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the M5 llama.cpp CUDA baseline")
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--llama-commit", required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument(
        "--golden", type=Path, default=Path("data/generated/tinyllama-chat-v1.0/reference.json")
    )
    parser.add_argument("--source-snapshot", type=Path, default=Path(".tlie-source-snapshot.json"))
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--power-mode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def request(port: int, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=180.0)
    payload = None if body is None else json.dumps(body)
    headers = {} if payload is None else {"Content-Type": "application/json"}
    connection.request(method, path, body=payload, headers=headers)
    response = connection.getresponse()
    content = response.read()
    status = response.status
    connection.close()
    if status != 200:
        raise RuntimeError(f"llama.cpp HTTP {status} from {path}: {content!r}")
    return json.loads(content)


def wait_ready(process: subprocess.Popen[str], port: int) -> None:
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, _ = process.communicate()
            raise RuntimeError(f"llama-server exited before ready: {stdout}")
        try:
            request(port, "GET", "/health")
            return
        except (ConnectionError, OSError, TimeoutError, RuntimeError):
            time.sleep(0.2)
    raise TimeoutError("llama-server did not become ready")


def repeated_context(seed: list[int], context: int) -> list[int]:
    if len(seed) < 2:
        raise ValueError("golden prompt must have at least two tokens")
    return [seed[0], *(seed[1 + index % (len(seed) - 1)] for index in range(context - 1))]


def completion_body(prompt: Any) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "n_predict": 32,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "repeat_penalty": 1.0,
        "seed": 0,
        "ignore_eos": True,
        "cache_prompt": False,
        "return_tokens": True,
    }


def validate_llama_checkout(server: Path, expected_commit: str) -> Path:
    resolved_server = server.resolve()
    try:
        source_root = resolved_server.parents[2]
    except IndexError as error:
        raise ValueError("llama.cpp server is not inside the expected build tree") from error
    expected_server = source_root / "build-cuda" / "bin" / "llama-server"
    if resolved_server != expected_server:
        raise ValueError("llama.cpp server is not inside <source>/build-cuda/bin")
    head = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_commit:
        raise ValueError(f"llama.cpp checkout commit {head} does not match {expected_commit}")
    dirty = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError("llama.cpp checkout has uncommitted source changes")
    return source_root


def validate_batch_samples(values: list[float], expected_count: int) -> list[float]:
    if len(values) != expected_count:
        raise ValueError(
            f"llama.cpp Batch 4 returned {len(values)} samples, expected {expected_count}"
        )
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("llama.cpp Batch 4 samples must be finite positive durations")
    return values


def main() -> None:
    args = parse_args()
    if not args.server.is_file() or not args.model.is_file():
        raise FileNotFoundError("llama.cpp server or GGUF model is missing")
    llama_source_root = validate_llama_checkout(args.server, args.llama_commit)
    context = 128
    output_tokens = 32
    warmup = 3
    samples = 10
    golden = cast(dict[str, Any], json.loads(args.golden.read_text(encoding="utf-8")))
    prompt = repeated_context(cast(list[int], golden["prompt_ids"]), context)
    reference = cast(dict[str, Any], json.loads(args.reference_report.read_text(encoding="utf-8")))
    if reference.get("engine") != "pytorch_fp16" or reference.get("status") != "available":
        raise ValueError("llama.cpp comparison requires a successful PyTorch reference report")
    expected_tokens = cast(list[int], reference["generated_tokens"])
    source_before = verify_snapshot(args.source_snapshot, Path.cwd())
    environment_dir = args.server.parent / "m5-llama-runtime"
    environment_dir.mkdir(parents=True, exist_ok=True)
    environment = cuda_env.prepare_cuda_tool_environment(environment_dir)
    command = environment.wrap_target(
        [
            str(args.server),
            "--model",
            str(args.model),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--ctx-size",
            "512",
            "--n-gpu-layers",
            "99",
            "--parallel",
            "4",
            "--cont-batching",
            "--metrics",
        ]
    )
    observations: list[dict[str, Any]] = []
    gpu_samples: list[dict[str, Any]] = []
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment.process_env,
    ) as server:
        try:
            wait_ready(server, args.port)
            gpu_samples.append(benchmark.capture_gpu_snapshot("llama_cpp_loaded"))
            for _ in range(warmup):
                request(args.port, "POST", "/completion", completion_body(prompt))
                request(args.port, "POST", "/completion", completion_body([prompt] * 4))
            for _ in range(samples):
                response = cast(
                    dict[str, Any],
                    request(args.port, "POST", "/completion", completion_body(prompt)),
                )
                timings = cast(dict[str, Any], response["timings"])
                tokens = cast(list[int], response["tokens"])
                predicted = int(timings["predicted_n"])
                if tokens != expected_tokens or predicted != output_tokens:
                    raise ValueError("llama.cpp generated tokens differ from golden")
                per_token_ms = float(timings["predicted_ms"]) / predicted
                observations.append(
                    {
                        "ttft_ms": float(timings["prompt_ms"]) + per_token_ms,
                        "tpot_ms": per_token_ms,
                        "output_tokens_per_second": output_tokens
                        * 1000.0
                        / (float(timings["prompt_ms"]) + float(timings["predicted_ms"])),
                    }
                )
                gpu_samples.append(benchmark.capture_gpu_snapshot("llama_cpp_sample"))
            batch_elapsed_samples: list[float] = []
            for _ in range(samples):
                batch_started = time.perf_counter()
                batch_response = cast(
                    list[dict[str, Any]],
                    request(args.port, "POST", "/completion", completion_body([prompt] * 4)),
                )
                batch_elapsed_samples.append(time.perf_counter() - batch_started)
                if len(batch_response) != 4 or any(
                    cast(list[int], response["tokens"]) != expected_tokens
                    for response in batch_response
                ):
                    raise ValueError("llama.cpp Batch 4 tokens differ from golden")
                gpu_samples.append(benchmark.capture_gpu_snapshot("llama_cpp_batch4_sample"))
        finally:
            server.terminate()
            try:
                server.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10.0)
    source_after = verify_snapshot(args.source_snapshot, Path.cwd())

    def median(name: str) -> float:
        return statistics.median(float(observation[name]) for observation in observations)

    memory_used_mib = max(float(sample["metrics"]["memory.used"]) for sample in gpu_samples)
    batch_elapsed_samples = validate_batch_samples(batch_elapsed_samples, samples)
    report = {
        "schema_version": 1,
        "engine": "llama_cpp",
        "status": "available",
        "dtype": "gguf_f16",
        "context": context,
        "output_tokens": output_tokens,
        "warmup": warmup,
        "samples": samples,
        "sampling": "greedy",
        "seed": 0,
        "prompt_seed": benchmark.PROMPT_SEED_TEXT,
        "ttft_ms_median": median("ttft_ms"),
        "tpot_ms_median": median("tpot_ms"),
        "output_tokens_per_second_median": median("output_tokens_per_second"),
        "peak_device_bytes": int(memory_used_mib * 1024**2),
        "peak_device_bytes_source": "nvidia-smi process-wide memory.used",
        "batch_4_total_tokens_per_second": statistics.median(
            4 * output_tokens / elapsed for elapsed in batch_elapsed_samples
        ),
        "batch_4_warmup": warmup,
        "batch_4_samples": samples,
        "tokens_match_reference": True,
        "generated_tokens": expected_tokens,
        "llama_cpp_commit": args.llama_commit,
        "llama_cpp_source_root": str(llama_source_root),
        "model_source_sha256": golden["source_model_sha256"],
        "model_gguf_sha256": sha256_file(args.model),
        "power_mode": args.power_mode,
        "gpu_monitor": benchmark.summarize_gpu_monitor("llama_cpp", gpu_samples),
        "source": {"before": source_before, "after": source_after},
        "note": "llama.cpp CUDA server; peak uses nvidia-smi and is not allocator-local",
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tokens_match": True}))


if __name__ == "__main__":
    main()
