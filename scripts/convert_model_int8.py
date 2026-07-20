from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from safetensors import safe_open

try:
    from model_assets import load_manifest, sha256_file, verify_file
    from tlie_format import write_mixed_tensor_file
except ModuleNotFoundError:
    from scripts.model_assets import load_manifest, sha256_file, verify_file
    from scripts.tlie_format import write_mixed_tensor_file


# Quantizing the token embedding and output projection made the fixed greedy
# corpus visibly repetitive even though the first-step cosine score remained
# high. Keeping these vocabulary-facing matrices in FP16 preserves the M3
# memory-reduction target while protecting token selection at both boundaries.
FP16_MATRIX_TENSORS = frozenset({"model.embed_tokens.weight", "lm_head.weight"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert pinned safetensors to per-output-channel W8A16 TLIEWGT"
    )
    parser.add_argument("--manifest", type=Path, default=Path("config/model_manifest.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/tinyllama-chat-v1.0"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--bootstrap-manifest",
        action="store_true",
        help="write an unpinned artifact once so its deterministic size/SHA can be added",
    )
    return parser.parse_args()


def quantize_per_output_channel(
    values: NDArray[np.float32],
) -> tuple[NDArray[np.int8], NDArray[np.float32]]:
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("INT8 weight-only quantization requires a non-empty rank-2 tensor")
    maximum = np.max(np.abs(values), axis=1).astype(np.float32, copy=False)
    scales = np.where(maximum == 0.0, np.float32(1.0), maximum / np.float32(127.0)).astype(
        np.float32, copy=False
    )
    quantized = np.clip(np.rint(values / scales[:, None]), -127, 127).astype(np.int8)
    return np.ascontiguousarray(quantized), np.ascontiguousarray(scales)


def quantized_tensors(path: Path, names: list[str]) -> Iterator[tuple[str, NDArray[np.generic]]]:
    with safe_open(path, framework="pt", device="cpu") as source:  # type: ignore[no-untyped-call]
        for name in names:
            tensor = source.get_tensor(name).float().contiguous().numpy()
            if tensor.ndim == 2 and name not in FP16_MATRIX_TENSORS:
                quantized, scales = quantize_per_output_channel(tensor)
                yield name, quantized
                yield f"{name}.scale", scales.astype("<f4", copy=False)
            elif tensor.ndim in (1, 2):
                yield name, np.ascontiguousarray(tensor.astype("<f2"))
            else:
                raise ValueError(f"unsupported TinyLlama tensor rank for {name}: {tensor.ndim}")


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    source_path = args.model_dir / "model.safetensors"
    verify_file(source_path, manifest["files"]["model.safetensors"])
    output = args.output or args.model_dir / "model-int8.tliewgt"
    with safe_open(source_path, framework="pt", device="cpu") as source:  # type: ignore[no-untyped-call]
        names = sorted(source.keys())
        rank_two_count = sum(
            source.get_tensor(name).ndim == 2 and name not in FP16_MATRIX_TENSORS for name in names
        )
    expected_count = 22 * 9 + 3
    if len(names) != expected_count:
        raise ValueError(f"expected {expected_count} tensors, found {len(names)}")
    record_count = len(names) + rank_two_count
    write_mixed_tensor_file(
        output,
        manifest["files"]["model.safetensors"]["sha256"],
        quantized_tensors(source_path, names),
        record_count,
    )
    sidecar = {
        "format": "TLIEWGT",
        "version": 1,
        "dtype": "int8_weight_only_per_output_channel",
        "activation_dtype": "float16",
        "scale_dtype": "float32",
        "source_model_sha256": manifest["files"]["model.safetensors"]["sha256"],
        "file_sha256": sha256_file(output),
        "file_size": output.stat().st_size,
        "model_tensor_count": len(names),
        "record_count": record_count,
        "quantized_tensor_count": rank_two_count,
        "fp16_matrix_tensors": sorted(FP16_MATRIX_TENSORS),
    }
    converted = cast(dict[str, object] | None, manifest.get("int8_converted_format"))
    if converted is None and not args.bootstrap_manifest:
        output.unlink(missing_ok=True)
        raise ValueError("manifest is missing int8_converted_format; bootstrap explicitly first")
    if converted is not None and not args.bootstrap_manifest:
        expected_sha256 = cast(str, converted["file_sha256"])
        expected_size = cast(int, converted["file_size"])
        if sidecar["file_sha256"] != expected_sha256 or sidecar["file_size"] != expected_size:
            output.unlink(missing_ok=True)
            raise ValueError("INT8 TLIEWGT does not match the pinned deterministic size/SHA-256")
    sidecar_path = output.with_suffix(output.suffix + ".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **sidecar}))


if __name__ == "__main__":
    main()
