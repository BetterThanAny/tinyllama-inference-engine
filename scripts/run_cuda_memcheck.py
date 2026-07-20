from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

if __package__:
    from scripts import wsl_cuda_env as cuda_env
else:
    import wsl_cuda_env as cuda_env  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CUDA target under a WSL-safe Compute Sanitizer memcheck"
    )
    parser.add_argument(
        "--target", type=Path, default=Path("build/cuda-release/tests/kernel_tests")
    )
    parser.add_argument("--tool", choices=("memcheck", "racecheck"), default="memcheck")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("target_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def require_tool(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise FileNotFoundError(
            f"{name} is required; do not install it globally without explicit approval"
        )
    return executable


def build_command(
    compute_sanitizer: str,
    environment: cuda_env.CudaToolEnvironment,
    target: Path,
    target_args: list[str],
    injection_path: Path | None = None,
    tool: str = "memcheck",
) -> list[str]:
    injection_arguments = (
        ["--injection-path", str(injection_path)] if injection_path is not None else []
    )
    tool_arguments = ["--tool", tool]
    if tool == "memcheck":
        tool_arguments.extend(["--leak-check", "full"])
    return [
        compute_sanitizer,
        *injection_arguments,
        "--launch-timeout",
        "60",
        "--target-processes",
        "all",
        *tool_arguments,
        "--error-exitcode",
        "99",
        *environment.wrap_target([str(target), *target_args]),
    ]


def require_clean_sanitizer(output: str, tool: str) -> None:
    if "before first instrumented API call" in output or "No attachable process found" in output:
        raise RuntimeError("Compute Sanitizer did not instrument the target CUDA process")
    if "Target application returned an error" in output:
        raise RuntimeError("Compute Sanitizer target application returned an error")
    if "process didn't terminate successfully" in output:
        raise RuntimeError("Compute Sanitizer target process did not terminate successfully")
    if tool == "memcheck":
        if re.search(r"LEAK SUMMARY:\s+0 bytes leaked in 0 allocations(?:\s|$)", output) is None:
            raise RuntimeError("Compute Sanitizer did not report a zero-leak LEAK SUMMARY")
        if re.search(r"ERROR SUMMARY:\s+0 errors(?:\s|$)", output) is None:
            raise RuntimeError("Compute Sanitizer did not report 'ERROR SUMMARY: 0 errors'")
        return
    if (
        re.search(r"RACECHECK SUMMARY:\s+0 hazards displayed \(0 errors, 0 warnings\)", output)
        is None
    ):
        raise RuntimeError("Compute Sanitizer did not report a clean RACECHECK SUMMARY")


def normalize_target_args(arguments: list[str]) -> list[str]:
    return arguments[1:] if arguments[:1] == ["--"] else arguments


def main() -> None:
    args = parse_args()
    target_args = normalize_target_args(args.target_args)
    if not args.target.is_file():
        raise FileNotFoundError(f"CUDA memcheck target does not exist: {args.target}")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("benchmarks/profiles") / f"{timestamp}-memcheck"
    output_dir.mkdir(parents=True, exist_ok=False)
    environment = cuda_env.prepare_cuda_tool_environment(output_dir)
    process_env = environment.process_env.copy()
    process_env.setdefault("NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE", "uds")
    compute_sanitizer = require_tool("compute-sanitizer")
    executable_injection_path = Path(compute_sanitizer).resolve().parent
    system_injection_path = Path("/usr/lib/nvidia-cuda-toolkit/compute-sanitizer")
    injection_path = next(
        (
            path
            for path in (executable_injection_path, system_injection_path)
            if (path / "libsanitizer-collection.so").is_file()
        ),
        None,
    )
    command = build_command(
        compute_sanitizer,
        environment,
        args.target,
        target_args,
        injection_path,
        args.tool,
    )
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=process_env,
    )
    log = completed.stdout
    log_path = output_dir / "compute-sanitizer.log"
    log_path.write_text(log, encoding="utf-8")
    print(log, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    require_clean_sanitizer(log, args.tool)
    print(output_dir)


if __name__ == "__main__":
    main()
