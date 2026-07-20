from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
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
    parser = argparse.ArgumentParser(description="Benchmark the M5 PyTorch FP16 baseline")
    parser.add_argument("--model-dir", type=Path, default=Path("models/tinyllama-chat-v1.0"))
    parser.add_argument("--manifest", type=Path, default=Path("config/model_manifest.json"))
    parser.add_argument("--source-snapshot", type=Path, default=Path(".tlie-source-snapshot.json"))
    parser.add_argument("--power-mode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-env-ready", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def repeated_context(seed: list[int], context: int) -> list[int]:
    if len(seed) < 2:
        raise ValueError("prompt seed must encode to at least two tokens")
    return [seed[0], *(seed[1 + (index % (len(seed) - 1))] for index in range(context - 1))]


def percentile(values: list[float], fraction: float) -> float:
    return float(benchmark.percentile(values, fraction))


def main() -> None:
    args = parse_args()
    if not args.cuda_env_ready:
        environment_dir = args.output.parent / "m5-pytorch-runtime"
        environment_dir.mkdir(parents=True, exist_ok=True)
        environment = cuda_env.prepare_cuda_tool_environment(environment_dir)
        if environment.target_prefix:
            completed = subprocess.run(
                environment.wrap_target([sys.executable, *sys.argv, "--cuda-env-ready"]),
                check=False,
                env=environment.process_env,
            )
            raise SystemExit(completed.returncode)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    context = 128
    output_tokens = 32
    warmup = 3
    samples = 10
    source_before = verify_snapshot(args.source_snapshot, Path.cwd())
    manifest = cast(dict[str, Any], json.loads(args.manifest.read_text(encoding="utf-8")))
    source_sha256 = str(manifest["files"]["model.safetensors"]["sha256"])
    if sha256_file(args.model_dir / "model.safetensors") != source_sha256:
        raise ValueError("PyTorch source model checksum differs from the pinned manifest")
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_dir, local_files_only=True, use_fast=False
    )
    seed = cast(list[int], tokenizer.encode(benchmark.PROMPT_SEED_TEXT))
    prompt = repeated_context(seed, context)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    gpu_samples = [benchmark.capture_gpu_snapshot("before_pytorch_benchmark")]

    def run(batch: int) -> dict[str, Any]:
        input_ids = torch.tensor([prompt] * batch, dtype=torch.long, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            prefill = model(input_ids=input_ids, use_cache=True)
            first = torch.argmax(prefill.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize()
        first_at = time.perf_counter()
        generated = [first]
        past = prefill.past_key_values
        previous = first
        with torch.inference_mode():
            for _ in range(1, output_tokens):
                decoded = model(input_ids=previous, past_key_values=past, use_cache=True)
                previous = torch.argmax(decoded.logits[:, -1, :], dim=-1, keepdim=True)
                generated.append(previous)
                past = decoded.past_key_values
        torch.cuda.synchronize()
        finished = time.perf_counter()
        tokens = torch.cat(generated, dim=1).cpu().tolist()
        return {
            "ttft_ms": (first_at - started) * 1000.0,
            "tpot_ms": (finished - first_at) * 1000.0 / (output_tokens - 1),
            "output_tokens_per_second": batch * output_tokens / (finished - started),
            "peak_device_bytes": int(torch.cuda.max_memory_allocated()),
            "tokens": tokens,
        }

    for _ in range(warmup):
        run(1)
        gpu_samples.append(benchmark.capture_gpu_snapshot("pytorch_warmup"))
    observations = []
    for _ in range(samples):
        observations.append(run(1))
        gpu_samples.append(benchmark.capture_gpu_snapshot("pytorch_batch1_sample"))
    batch_observations = []
    for _ in range(samples):
        batch_observations.append(run(4))
        gpu_samples.append(benchmark.capture_gpu_snapshot("pytorch_batch4_sample"))
    expected_tokens = cast(list[int], observations[0]["tokens"][0])
    token_match = len(expected_tokens) == output_tokens and all(
        observation["tokens"][0] == expected_tokens for observation in observations
    )
    batch_match = all(
        all(tokens == expected_tokens for tokens in observation["tokens"])
        for observation in batch_observations
    )
    if not token_match or not batch_match:
        raise ValueError("PyTorch generated tokens differ from the fixed golden tokens")
    gpu_samples.append(benchmark.capture_gpu_snapshot("after_pytorch_benchmark"))
    gpu_monitor = benchmark.summarize_gpu_monitor("pytorch", gpu_samples)
    gpu_monitor["poll_interval_seconds"] = None
    gpu_monitor["sampling_strategy"] = "before/after and after every warmup or measured workload"
    source_after = verify_snapshot(args.source_snapshot, Path.cwd())

    def values(name: str) -> list[float]:
        return [float(observation[name]) for observation in observations]

    batch_tps = [
        float(observation["output_tokens_per_second"]) for observation in batch_observations
    ]
    report = {
        "schema_version": 1,
        "engine": "pytorch_fp16",
        "status": "available",
        "dtype": "float16",
        "context": context,
        "output_tokens": output_tokens,
        "warmup": warmup,
        "samples": samples,
        "sampling": "greedy",
        "seed": 0,
        "prompt_seed": benchmark.PROMPT_SEED_TEXT,
        "model_source_sha256": source_sha256,
        "ttft_ms_median": statistics.median(values("ttft_ms")),
        "ttft_ms_p95": percentile(values("ttft_ms"), 0.95),
        "tpot_ms_median": statistics.median(values("tpot_ms")),
        "tpot_ms_p95": percentile(values("tpot_ms"), 0.95),
        "output_tokens_per_second_median": statistics.median(values("output_tokens_per_second")),
        "peak_device_bytes": max(
            int(observation["peak_device_bytes"]) for observation in observations
        ),
        "batch_4_total_tokens_per_second": statistics.median(batch_tps),
        "tokens_deterministic": token_match and batch_match,
        "generated_tokens": expected_tokens,
        "power_mode": args.power_mode,
        "gpu_before": gpu_samples[0]["metrics"],
        "gpu_after": gpu_samples[-1]["metrics"],
        "gpu_monitor": gpu_monitor,
        "source": {"before": source_before, "after": source_after},
        "note": "Hugging Face Transformers eager CUDA reference; deterministic across samples",
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tokens_match": True}))


if __name__ == "__main__":
    main()
