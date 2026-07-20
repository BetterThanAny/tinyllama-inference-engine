from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from model_assets import load_manifest, verify_file
from transformers import AutoTokenizer

CASES = [
    "",
    "Hello, TinyLlama!",
    "The capital of France is",
    "你好，TinyLlama！",  # noqa: RUF001 - intentional Unicode punctuation coverage
    "line one\nline two\tend",
    "emoji: 🦙🚀",
    " leading and trailing ",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed tokenizer corpus from Transformers")
    parser.add_argument("--manifest", type=Path, default=Path("config/model_manifest.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/tinyllama-chat-v1.0"))
    parser.add_argument("--output", type=Path, default=Path("data/golden/tokenizer_cases.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    verify_file(args.model_dir / "tokenizer.model", manifest["files"]["tokenizer.model"])
    tokenizer: Any = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_dir, local_files_only=True, use_fast=False, legacy=False
    )
    cases = []
    for text in CASES:
        ids = tokenizer.encode(text, add_special_tokens=True)
        cases.append({"text": text, "add_bos": True, "add_eos": False, "ids": ids})
    document = {
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "tokenizer_sha256": manifest["files"]["tokenizer.model"]["sha256"],
        "reference": "transformers.AutoTokenizer(use_fast=False, legacy=False)",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    print(f"wrote {len(cases)} tokenizer golden cases to {args.output}")


if __name__ == "__main__":
    main()
