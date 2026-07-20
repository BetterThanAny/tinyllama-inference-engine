#!/usr/bin/env bash

set -euo pipefail

host="${TLIE_WSL_HOST:-my-wsl}"
remote_dir="${TLIE_WSL_DIR:-tinyllama-inference-engine}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${host}" ]]; then
  echo "TLIE_WSL_HOST must not be empty" >&2
  exit 2
fi

if [[ -z "${remote_dir}" || "${remote_dir}" == /* || ! "${remote_dir}" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  echo "TLIE_WSL_DIR must be a safe path relative to the remote home directory" >&2
  exit 2
fi

case "/${remote_dir}/" in
  */../* | */./* | *//* )
    echo "TLIE_WSL_DIR must not contain '.', '..', or empty path components" >&2
    exit 2
    ;;
esac

(
  cd "${repo_root}"
  mise exec -- python scripts/source_snapshot.py \
    --root "${repo_root}" \
    --output "${repo_root}/.tlie-source-snapshot.json"
)

ssh -o BatchMode=yes -- "${host}" "mkdir -p -- '${remote_dir}'"

rsync \
  --archive \
  --compress \
  --delete \
  --itemize-changes \
  --exclude '/.git/' \
  --exclude '.DS_Store' \
  --exclude '/build/' \
  --exclude '/.cache/' \
  --exclude '/.venv/' \
  --exclude '.mypy_cache/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '__pycache__/' \
  --exclude '/models/' \
  --exclude '/data/generated/' \
  --exclude '/benchmarks/results/' \
  --exclude '/benchmarks/profiles/' \
  "${repo_root}/" \
  "${host}:${remote_dir}/"

echo "Synced ${repo_root} to ${host}:~/${remote_dir}"
