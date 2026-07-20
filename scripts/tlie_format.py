from __future__ import annotations

import hashlib
import os
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

MAGIC = b"TLIEWGT\0"
VERSION = 1
DTYPE_FLOAT32 = 1
DTYPE_FLOAT16 = 2
DTYPE_INT8 = 3
ALIGNMENT = 64


def _padding(offset: int) -> int:
    return (ALIGNMENT - offset % ALIGNMENT) % ALIGNMENT


def write_tensor_file(
    path: Path,
    source_sha256: str,
    tensors: Iterable[tuple[str, NDArray[np.float32]]],
    tensor_count: int,
    storage_dtype: Literal["float32", "float16"] = "float32",
) -> None:
    source_digest = bytes.fromhex(source_sha256)
    if len(source_digest) != 32:
        raise ValueError("source_sha256 must contain exactly 32 bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    numpy_dtype = "<f4" if storage_dtype == "float32" else "<f2"
    dtype_code = DTYPE_FLOAT32 if storage_dtype == "float32" else DTYPE_FLOAT16
    try:
        with temporary.open("wb") as output:
            output.write(struct.pack("<8sII32s", MAGIC, VERSION, tensor_count, source_digest))
            written = 0
            for name, tensor in tensors:
                encoded_name = name.encode("utf-8")
                if not encoded_name or len(encoded_name) > 4096:
                    raise ValueError(f"invalid tensor name: {name!r}")
                array = np.asarray(tensor, dtype=numpy_dtype, order="C")
                if not array.shape or any(dimension <= 0 for dimension in array.shape):
                    raise ValueError(f"tensor {name} must have a non-empty shape")
                payload = memoryview(array).cast("B")
                output.write(
                    struct.pack(
                        "<HBBQ32s",
                        len(encoded_name),
                        dtype_code,
                        array.ndim,
                        payload.nbytes,
                        hashlib.sha256(payload).digest(),
                    )
                )
                output.write(struct.pack(f"<{array.ndim}Q", *array.shape))
                output.write(encoded_name)
                output.write(b"\0" * _padding(output.tell()))
                output.write(payload)
                written += 1
            if written != tensor_count:
                raise ValueError(f"tensor count mismatch: expected {tensor_count}, wrote {written}")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_mixed_tensor_file(
    path: Path,
    source_sha256: str,
    tensors: Iterable[tuple[str, NDArray[np.generic]]],
    tensor_count: int,
) -> None:
    source_digest = bytes.fromhex(source_sha256)
    if len(source_digest) != 32:
        raise ValueError("source_sha256 must contain exactly 32 bytes")
    dtype_records = {
        np.dtype("<f4"): DTYPE_FLOAT32,
        np.dtype("<f2"): DTYPE_FLOAT16,
        np.dtype("i1"): DTYPE_INT8,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as output:
            output.write(struct.pack("<8sII32s", MAGIC, VERSION, tensor_count, source_digest))
            written = 0
            for name, tensor in tensors:
                encoded_name = name.encode("utf-8")
                if not encoded_name or len(encoded_name) > 4096:
                    raise ValueError(f"invalid tensor name: {name!r}")
                array = np.ascontiguousarray(tensor)
                dtype_code = dtype_records.get(array.dtype)
                if dtype_code is None:
                    raise ValueError(f"unsupported mixed tensor dtype for {name}: {array.dtype}")
                if not array.shape or any(dimension <= 0 for dimension in array.shape):
                    raise ValueError(f"tensor {name} must have a non-empty shape")
                payload = memoryview(array).cast("B")
                output.write(
                    struct.pack(
                        "<HBBQ32s",
                        len(encoded_name),
                        dtype_code,
                        array.ndim,
                        payload.nbytes,
                        hashlib.sha256(payload).digest(),
                    )
                )
                output.write(struct.pack(f"<{array.ndim}Q", *array.shape))
                output.write(encoded_name)
                output.write(b"\0" * _padding(output.tell()))
                output.write(payload)
                written += 1
            if written != tensor_count:
                raise ValueError(f"tensor count mismatch: expected {tensor_count}, wrote {written}")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
