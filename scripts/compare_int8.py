from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

try:
    import wsl_cuda_env as cuda_env
    from compare_logits import read_golden_logits, resolve_reference, run_engine
except ModuleNotFoundError:
    from scripts import wsl_cuda_env as cuda_env
    from scripts.compare_logits import read_golden_logits, resolve_reference, run_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare W8A16 against FP16 and FP32 references")
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/generated/tinyllama-chat-v1.0/reference.json"),
    )
    parser.add_argument("--engine", type=Path, default=Path("build/cuda-release/tinyllama_cuda"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/tinyllama-chat-v1.0"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-cosine", type=float, default=0.98)
    return parser.parse_args()


def softmax(values: NDArray[np.float32]) -> NDArray[np.float64]:
    shifted = values.astype(np.float64) - float(np.max(values))
    exponent = np.exp(shifted)
    return cast(NDArray[np.float64], exponent / np.sum(exponent))


def distribution_metrics(
    actual: NDArray[np.float32], reference: NDArray[np.float32]
) -> dict[str, float | int | bool]:
    if actual.shape != reference.shape or actual.ndim != 1 or not np.isfinite(actual).all():
        raise ValueError("logits must be same-shape finite vectors")
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    norm_product = float(np.linalg.norm(actual) * np.linalg.norm(reference))
    cosine = float(np.dot(actual.astype(np.float64), reference.astype(np.float64)) / norm_product)
    actual_probability = softmax(actual)
    reference_probability = softmax(reference)
    midpoint = (actual_probability + reference_probability) / 2.0
    epsilon = np.finfo(np.float64).tiny
    js_divergence = 0.5 * float(
        np.sum(reference_probability * np.log((reference_probability + epsilon) / midpoint))
        + np.sum(actual_probability * np.log((actual_probability + epsilon) / midpoint))
    )
    reference_top = int(np.argmax(reference))
    reference_top_nll = -math.log(float(reference_probability[reference_top]) + epsilon)
    actual_top_nll = -math.log(float(actual_probability[reference_top]) + epsilon)
    return {
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "cosine_similarity": cosine,
        "top1_token": int(np.argmax(actual)),
        "reference_top1_token": reference_top,
        "top1_matches": int(np.argmax(actual)) == reference_top,
        "jensen_shannon_divergence": js_divergence,
        "reference_top1_nll": reference_top_nll,
        "actual_probability_reference_top1_nll": actual_top_nll,
        "top1_perplexity_ratio_proxy": math.exp(actual_top_nll - reference_top_nll),
    }


def common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        count += 1
    return count


def parse_engine_json(completed: Any, operation: str) -> dict[str, Any]:
    if completed.returncode != 0:
        raise RuntimeError(
            f"{operation} failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    return cast(dict[str, Any], json.loads(completed.stdout))


def main() -> None:
    args = parse_args()
    reference_path = resolve_reference(args.reference)
    reference = cast(dict[str, Any], json.loads(reference_path.read_text(encoding="utf-8")))
    prompt = cast(str, reference["prompt"])
    golden_tokens = cast(list[int], reference["greedy_tokens"])
    fp32_logits = read_golden_logits(reference_path, reference)
    with tempfile.TemporaryDirectory(prefix="tlie-compare-int8-") as temporary_directory:
        environment = cuda_env.prepare_cuda_tool_environment(Path(temporary_directory))

        def execute(command: str, *arguments: str) -> dict[str, Any]:
            return parse_engine_json(
                run_engine(
                    [str(args.engine), command, str(args.model_dir), *arguments], environment
                ),
                command,
            )

        fp16_logits_result = execute("logits", prompt)
        int8_logits_result = execute("logits-int8", prompt)
        fp16_generation = execute("generate", prompt, str(len(golden_tokens)))
        int8_generation = execute("generate-int8", prompt, str(len(golden_tokens)))

    fp16_logits = np.asarray(cast(list[float], fp16_logits_result["logits"]), dtype=np.float32)
    int8_logits = np.asarray(cast(list[float], int8_logits_result["logits"]), dtype=np.float32)
    fp16_tokens = cast(list[int], fp16_generation["generated_tokens"])
    int8_tokens = cast(list[int], int8_generation["generated_tokens"])
    int8_metrics = distribution_metrics(int8_logits, fp32_logits)
    fp16_metrics = distribution_metrics(fp16_logits, fp32_logits)
    prefix = common_prefix(int8_tokens, golden_tokens)
    agreement = sum(
        actual == expected for actual, expected in zip(int8_tokens, golden_tokens, strict=False)
    ) / len(golden_tokens)
    passed = (
        bool(int8_metrics["top1_matches"])
        and float(int8_metrics["cosine_similarity"]) >= args.minimum_cosine
        and int8_tokens == golden_tokens
        and fp16_tokens == golden_tokens
    )
    report = {
        "schema_version": 1,
        "reference": str(reference_path),
        "prompt": prompt,
        "compared_logits": int(fp32_logits.size),
        "compared_tokens": len(golden_tokens),
        "fp16_vs_fp32": fp16_metrics,
        "int8_vs_fp32": int8_metrics,
        "generation": {
            "golden_tokens": golden_tokens,
            "fp16_tokens": fp16_tokens,
            "int8_tokens": int8_tokens,
            "int8_common_prefix_tokens": prefix,
            "int8_token_position_agreement": agreement,
            "fp16_text": fp16_generation["text"],
            "int8_text": int8_generation["text"],
        },
        "acceptance": {
            "passed": passed,
            "minimum_cosine": args.minimum_cosine,
            "requires_fp32_top1_match": True,
            "requires_fp16_exact_golden_tokens": True,
            "requires_int8_exact_golden_tokens": True,
        },
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
