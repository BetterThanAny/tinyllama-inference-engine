from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict, cast
from urllib.request import Request, urlopen


class FileRecord(TypedDict):
    sha256: str
    size: int


class ModelManifest(TypedDict):
    schema_version: int
    model_id: str
    revision: str
    license: str
    files: dict[str, FileRecord]
    converted_format: dict[str, object]
    cuda_converted_format: dict[str, object]


def load_manifest(path: Path) -> ModelManifest:
    with path.open(encoding="utf-8") as handle:
        value = cast(dict[str, object], json.load(handle))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported model manifest schema")
    return cast(ModelManifest, value)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, record: FileRecord) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required model asset is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != record["size"]:
        raise ValueError(
            f"size mismatch for {path.name}: expected {record['size']}, got {actual_size}"
        )
    actual_checksum = sha256_file(path)
    if actual_checksum != record["sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {record['sha256']}, got {actual_checksum}"
        )


def download_file(url: str, destination: Path, record: FileRecord) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_file(destination, record)
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    request = Request(url, headers={"User-Agent": "tlie-model-preparer/0.1"})
    digest = hashlib.sha256()
    received = 0
    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                received += len(chunk)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if received != record["size"] or digest.hexdigest() != record["sha256"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"download verification failed for {destination.name}: "
            f"size={received}, sha256={digest.hexdigest()}"
        )
    temporary.replace(destination)
