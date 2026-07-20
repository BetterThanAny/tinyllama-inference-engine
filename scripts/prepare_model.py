from __future__ import annotations

import argparse
from pathlib import Path

from model_assets import download_file, load_manifest, verify_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify pinned TinyLlama assets")
    parser.add_argument(
        "--manifest", type=Path, default=Path("config/model_manifest.json"), help="model manifest"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("models/tinyllama-chat-v1.0"), help="asset directory"
    )
    parser.add_argument(
        "--include-weights", action="store_true", help="also download the 2.2 GB safetensors file"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    selected = list(manifest["files"])
    if not args.include_weights:
        selected.remove("model.safetensors")
    base_url = f"https://huggingface.co/{manifest['model_id']}/resolve/{manifest['revision']}"
    for filename in selected:
        record = manifest["files"][filename]
        destination = args.output / filename
        download_file(f"{base_url}/{filename}", destination, record)
        verify_file(destination, record)
        print(f"verified {filename}: {record['sha256']}")


if __name__ == "__main__":
    main()
