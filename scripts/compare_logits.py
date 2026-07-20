from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

if __package__:
    from scripts import wsl_cuda_env as cuda_env
    from scripts.model_assets import sha256_file
else:
    import wsl_cuda_env as cuda_env  # type: ignore[no-redef]
    from model_assets import sha256_file  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FP16 CUDA greedy output with the fixed FP32 golden trace"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/generated/tinyllama-chat-v1.0/reference.json"),
    )
    parser.add_argument("--engine", type=Path, default=Path("build/cuda-release/tinyllama_cuda"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/tinyllama-chat-v1.0"))
    parser.add_argument("--atol", type=float, default=0.15)
    parser.add_argument("--rtol", type=float, default=0.03)
    return parser.parse_args()


def resolve_reference(path: Path) -> Path:
    candidate = path / "reference.json" if path.is_dir() else path
    if not candidate.is_file():
        raise FileNotFoundError(f"fixed golden metadata is missing: {candidate}")
    return candidate


def read_golden_logits(metadata_path: Path, metadata: dict[str, Any]) -> NDArray[np.float32]:
    trace_path = metadata_path.parent / cast(str, metadata["trace_file"])
    if sha256_file(trace_path) != cast(str, metadata["trace_sha256"]):
        raise ValueError("golden trace does not match its externally pinned SHA-256")
    contents = trace_path.read_bytes()
    if len(contents) < 48:
        raise ValueError("golden trace header is truncated")
    magic, version, count, source_digest = struct.unpack_from("<8sII32s", contents)
    if magic != b"TLIEWGT\0" or version != 1 or count <= 0:
        raise ValueError("golden trace header is invalid")
    if source_digest.hex() != cast(str, metadata["source_model_sha256"]):
        raise ValueError("golden trace source checksum is invalid")
    offset = 48
    logits: NDArray[np.float32] | None = None
    for _ in range(count):
        if offset + 44 > len(contents):
            raise ValueError("golden trace tensor header is truncated")
        name_length, dtype, rank, byte_count, checksum = struct.unpack_from(
            "<HBBQ32s", contents, offset
        )
        offset += 44
        if dtype != 1 or rank <= 0 or offset + rank * 8 + name_length > len(contents):
            raise ValueError("golden trace tensor metadata is invalid")
        shape = struct.unpack_from(f"<{rank}Q", contents, offset)
        offset += rank * 8
        name = contents[offset : offset + name_length].decode("utf-8")
        offset += name_length
        offset += (64 - offset % 64) % 64
        payload = contents[offset : offset + byte_count]
        if len(payload) != byte_count or hashlib.sha256(payload).digest() != checksum:
            raise ValueError(f"golden trace tensor checksum failed: {name}")
        expected_bytes = math.prod(shape) * 4
        if expected_bytes != byte_count:
            raise ValueError(f"golden trace tensor shape is invalid: {name}")
        if name == "logits":
            logits = np.frombuffer(payload, dtype="<f4").copy()
        offset += byte_count
    if offset != len(contents) or logits is None:
        raise ValueError("golden trace has trailing bytes or no logits tensor")
    return logits


def run_engine(
    command: list[str], environment: cuda_env.CudaToolEnvironment
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        environment.wrap_target(command),
        check=False,
        capture_output=True,
        text=True,
        env=environment.process_env,
    )


def main() -> None:
    args = parse_args()
    reference_path = resolve_reference(args.reference)
    reference = cast(dict[str, Any], json.loads(reference_path.read_text(encoding="utf-8")))
    prompt = cast(str, reference["prompt"])
    expected = cast(list[int], reference["greedy_tokens"])
    golden_logits = read_golden_logits(reference_path, reference)
    with tempfile.TemporaryDirectory(prefix="tlie-compare-logits-") as temporary_directory:
        environment = cuda_env.prepare_cuda_tool_environment(Path(temporary_directory))
        logits_command = [str(args.engine), "logits", str(args.model_dir), prompt]
        logits_completed = run_engine(logits_command, environment)
        if logits_completed.returncode != 0:
            raise RuntimeError(
                f"CUDA logits comparison failed ({logits_completed.returncode}): "
                f"{logits_completed.stderr.strip()}"
            )
        logits_result = cast(dict[str, Any], json.loads(logits_completed.stdout))
        actual_logits = np.asarray(cast(list[float], logits_result["logits"]), dtype=np.float32)
        if actual_logits.shape != golden_logits.shape or not np.isfinite(actual_logits).all():
            raise ValueError("CUDA logits have an invalid shape or contain NaN/Inf")
        difference = np.abs(actual_logits - golden_logits)
        tolerance = args.atol + args.rtol * np.abs(golden_logits)
        if not np.all(difference <= tolerance):
            index = int(np.argmax(difference - tolerance))
            raise AssertionError(
                f"CUDA logit {index} is outside tolerance: actual={actual_logits[index]}, "
                f"expected={golden_logits[index]}, abs_error={difference[index]}, "
                f"allowed={tolerance[index]}"
            )

        generate_command = [
            str(args.engine),
            "generate",
            str(args.model_dir),
            prompt,
            str(len(expected)),
            str(reference_path),
        ]
        completed = run_engine(generate_command, environment)
        if completed.returncode != 0:
            raise RuntimeError(
                f"CUDA golden comparison failed ({completed.returncode}): "
                f"{completed.stderr.strip()}"
            )
    result = cast(dict[str, Any], json.loads(completed.stdout))
    actual = cast(list[int], result["generated_tokens"])
    if actual != expected:
        mismatch = next(
            (
                index
                for index, pair in enumerate(zip(actual, expected, strict=False))
                if pair[0] != pair[1]
            ),
            min(len(actual), len(expected)),
        )
        raise AssertionError(f"greedy token mismatch at position {mismatch}")
    if not isinstance(result.get("text"), str):
        raise TypeError("CUDA engine response is missing decoded text")
    print(
        json.dumps(
            {
                "status": "passed",
                "compared_tokens": len(expected),
                "compared_logits": int(golden_logits.size),
                "logits_atol": args.atol,
                "logits_rtol": args.rtol,
                "logits_max_abs_error": float(difference.max()),
                "reference": str(reference_path),
                "engine": str(args.engine),
            }
        )
    )


if __name__ == "__main__":
    main()
