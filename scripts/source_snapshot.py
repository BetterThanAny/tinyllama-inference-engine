from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast

SNAPSHOT_NAME = ".tlie-source-snapshot.json"
EXCLUDED_ROOTS = {".cache", ".git", ".venv", "build", "models"}
EXCLUDED_SUBTREES = {"benchmarks/profiles", "benchmarks/results", "data/generated"}
EXCLUDED_DIRECTORY_NAMES = {".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}


class FileRecord(TypedDict):
    size: int
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the deterministic Mac source snapshot used by remote CUDA reports"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path(SNAPSHOT_NAME))
    return parser.parse_args()


def _is_excluded(relative: Path) -> bool:
    parts = relative.parts
    if not parts:
        return False
    if parts[0] in EXCLUDED_ROOTS:
        return True
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in parts[:-1]):
        return True
    if relative.as_posix() == SNAPSHOT_NAME or relative.name == ".DS_Store":
        return True
    return any(
        relative == Path(subtree) or Path(subtree) in relative.parents
        for subtree in EXCLUDED_SUBTREES
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root: Path) -> dict[str, FileRecord]:
    root = root.resolve(strict=True)
    records: dict[str, FileRecord] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if _is_excluded(relative) or path.is_dir():
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source snapshot accepts only regular files: {relative.as_posix()}")
        records[relative.as_posix()] = {"size": path.stat().st_size, "sha256": _sha256_file(path)}
    if not records:
        raise ValueError("source snapshot contains 0 files")
    return records


def tree_sha256(files: dict[str, FileRecord]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        record = files[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def git_metadata(root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--short", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": head.stdout.strip() if head.returncode == 0 else None,
        "working_tree_dirty": status.returncode != 0 or bool(status.stdout.strip()),
    }


def create_snapshot(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    files = collect_files(root)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_root_name": root.name,
        "git": git_metadata(root),
        "tree_sha256": tree_sha256(files),
        "files": files,
    }


def verify_snapshot(path: Path, root: Path) -> dict[str, Any]:
    document = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if document.get("schema_version") != 1:
        raise ValueError("unsupported source snapshot schema")
    expected_files = document.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("source snapshot contains 0 file records")
    actual_files = collect_files(root)
    if actual_files != expected_files:
        missing = sorted(set(expected_files) - set(actual_files))
        unexpected = sorted(set(actual_files) - set(expected_files))
        changed = sorted(
            name
            for name in set(actual_files) & set(expected_files)
            if actual_files[name] != expected_files[name]
        )
        raise ValueError(
            "source mirror differs from the Mac snapshot: "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )
    actual_tree = tree_sha256(actual_files)
    if document.get("tree_sha256") != actual_tree:
        raise ValueError("source snapshot tree SHA-256 is invalid")
    git = document.get("git")
    if not isinstance(git, dict) or not isinstance(git.get("working_tree_dirty"), bool):
        raise ValueError("source snapshot Git metadata is invalid")
    commit = git.get("commit")
    if commit is not None and (not isinstance(commit, str) or len(commit) != 40):
        raise ValueError("source snapshot Git commit is invalid")
    return document


def main() -> None:
    args = parse_args()
    root = args.root.resolve(strict=True)
    output = args.output if args.output.is_absolute() else root / args.output
    document = create_snapshot(root)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "snapshot": str(output),
                "files": len(cast(dict[str, Any], document["files"])),
                "tree_sha256": document["tree_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
