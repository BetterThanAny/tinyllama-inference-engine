from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from model_assets import load_manifest, sha256_file, verify_file
from numpy.typing import NDArray
from safetensors import safe_open
from tlie_format import write_tensor_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pinned safetensors to TLIEWGT FP32")
    parser.add_argument("--manifest", type=Path, default=Path("config/model_manifest.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/tinyllama-chat-v1.0"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def tensors_from_safetensors(path: Path, names: list[str]) -> Iterator[tuple[str, NDArray[Any]]]:
    with safe_open(path, framework="pt", device="cpu") as source:  # type: ignore[no-untyped-call]
        for name in names:
            tensor = source.get_tensor(name)
            yield name, tensor.float().contiguous().numpy()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    source_path = args.model_dir / "model.safetensors"
    verify_file(source_path, manifest["files"]["model.safetensors"])
    output = args.output or args.model_dir / "model-fp32.tliewgt"
    with safe_open(source_path, framework="pt", device="cpu") as source:  # type: ignore[no-untyped-call]
        names = sorted(source.keys())
    expected_count = 22 * 9 + 3
    if len(names) != expected_count:
        raise ValueError(f"expected {expected_count} tensors, found {len(names)}")
    write_tensor_file(
        output,
        manifest["files"]["model.safetensors"]["sha256"],
        tensors_from_safetensors(source_path, names),
        len(names),
    )
    sidecar = {
        "format": "TLIEWGT",
        "version": 1,
        "dtype": "float32",
        "source_model_sha256": manifest["files"]["model.safetensors"]["sha256"],
        "file_sha256": sha256_file(output),
        "file_size": output.stat().st_size,
        "tensor_count": len(names),
    }
    converted = manifest["converted_format"]
    expected_file_sha256 = cast(str, converted["file_sha256"])
    expected_file_size = cast(int, converted["file_size"])
    if sidecar["file_sha256"] != expected_file_sha256 or sidecar["file_size"] != expected_file_size:
        output.unlink(missing_ok=True)
        raise ValueError(
            "converted TLIEWGT does not match the pinned deterministic size and SHA-256"
        )
    sidecar_path = output.with_suffix(output.suffix + ".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} ({sidecar['file_size']} bytes, sha256={sidecar['file_sha256']})")


if __name__ == "__main__":
    main()
