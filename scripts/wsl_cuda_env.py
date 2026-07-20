from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

WSL_LIB_DIR = Path("/usr/lib/wsl/lib")
WSL_LIBCUDA = WSL_LIB_DIR / "libcuda.so.1"
WSL_DRIVER_ROOT = Path("/usr/lib/wsl/drivers")
LINUX_LIBCUDA = Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1")
AUDIT_SOURCE = Path(__file__).with_name("wsl_cuda_ld_audit.c")


@dataclass(frozen=True)
class CudaToolEnvironment:
    process_env: dict[str, str]
    target_prefix: tuple[str, ...]
    audit_library: Path | None

    def wrap_target(self, command: list[str]) -> list[str]:
        return [*self.target_prefix, *command]


def is_wsl(release_path: Path = Path("/proc/sys/kernel/osrelease")) -> bool:
    try:
        return "microsoft" in release_path.read_text(encoding="utf-8").lower()
    except OSError:
        return False


def _has_conflicting_linux_driver() -> bool:
    if not LINUX_LIBCUDA.exists():
        return False
    try:
        return not LINUX_LIBCUDA.samefile(WSL_LIBCUDA)
    except OSError:
        return True


def prepare_cuda_tool_environment(
    output_dir: Path, environ: Mapping[str, str] | None = None
) -> CudaToolEnvironment:
    process_env = dict(os.environ if environ is None else environ)
    if not is_wsl():
        return CudaToolEnvironment(process_env, (), None)
    if not WSL_LIBCUDA.is_file():
        raise RuntimeError(f"WSL CUDA loader is missing at {WSL_LIBCUDA}")

    driver_directories = sorted(
        {path.parent for path in WSL_DRIVER_ROOT.glob("*/libnvidia-ptxjitcompiler.so.1")}
    )
    if len(driver_directories) != 1:
        raise RuntimeError(
            "expected exactly one active WSL NVIDIA driver directory containing "
            f"libnvidia-ptxjitcompiler.so.1, found {len(driver_directories)}"
        )

    library_entries = [str(driver_directories[0]), str(WSL_LIB_DIR)]
    inherited_library_path = process_env.get("LD_LIBRARY_PATH")
    if inherited_library_path:
        library_entries.append(inherited_library_path)
    library_path = ":".join(library_entries)
    process_env["LD_LIBRARY_PATH"] = library_path

    target_assignments = [f"LD_LIBRARY_PATH={library_path}"]
    audit_library: Path | None = None
    if _has_conflicting_linux_driver():
        compiler = shutil.which("cc")
        if compiler is None:
            raise RuntimeError("cc is required to build the WSL CUDA loader audit module")
        if not AUDIT_SOURCE.is_file():
            raise RuntimeError(f"WSL CUDA loader audit source is missing at {AUDIT_SOURCE}")
        audit_library = output_dir / "wsl_cuda_ld_audit.so"
        subprocess.run(
            [
                compiler,
                "-std=c11",
                "-fPIC",
                "-shared",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(AUDIT_SOURCE),
                "-o",
                str(audit_library),
            ],
            check=True,
            env=process_env,
        )
        target_assignments.insert(0, f"LD_AUDIT={audit_library.resolve()}")

    return CudaToolEnvironment(
        process_env,
        ("/usr/bin/env", *target_assignments),
        audit_library,
    )
