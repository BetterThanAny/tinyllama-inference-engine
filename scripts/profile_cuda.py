from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

if __package__:
    from scripts import wsl_cuda_env as cuda_env
else:
    import wsl_cuda_env as cuda_env  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture reproducible CUDA Nsight profiles")
    parser.add_argument("--tool", choices=["systems", "compute", "both"], default="both")
    parser.add_argument("--engine", type=Path, default=Path("build/cuda-release/tinyllama_cuda"))
    parser.add_argument(
        "--kernel-benchmarks",
        type=Path,
        default=Path("build/cuda-release/benchmarks/kernel_benchmarks"),
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/tinyllama-chat-v1.0"))
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--mode", choices=["float16", "int8"], default="float16")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def require_tool(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise FileNotFoundError(
            f"{name} is required for this profile; do not install it globally without approval"
        )
    return executable


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def run_capture(command: list[str], env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    return completed.stdout


def require_nonempty_artifact(path: Path, tool: str, raw_artifact: Path | None = None) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    raw_detail = ""
    if raw_artifact is not None and raw_artifact.exists():
        raw_detail = f"; raw capture exists at {raw_artifact}, but it is not an importable report"
    raise RuntimeError(f"{tool} did not produce a non-empty report at {path}{raw_detail}")


def parse_named_csv_rows(content: str) -> list[str]:
    accepted_name_columns = {"Kernel Name", "Name", "Range"}
    reader = csv.reader(content.splitlines())
    name_index: int | None = None
    names: list[str] = []
    for row in reader:
        stripped = [cell.strip() for cell in row]
        matching_indices = [
            index for index, value in enumerate(stripped) if value in accepted_name_columns
        ]
        if matching_indices:
            name_index = matching_indices[0]
            continue
        if name_index is not None and len(stripped) > name_index and stripped[name_index]:
            names.append(stripped[name_index])
    return names


def require_profile_rows(path: Path, tool: str, required_names: set[str] | None = None) -> None:
    require_nonempty_artifact(path, tool)
    names = parse_named_csv_rows(path.read_text(encoding="utf-8"))
    if not names:
        raise RuntimeError(f"{tool} reported 0 data rows in {path}")
    if required_names:
        missing = sorted(
            required
            for required in required_names
            if not any(name == required or name.endswith(f":{required}") for name in names)
        )
        if missing:
            raise RuntimeError(f"{tool} is missing required ranges: {', '.join(missing)}")


def capture_nsys_stats(
    nsys: str,
    report: Path,
    report_name: str,
    output: Path,
    env: dict[str, str],
) -> None:
    content = run_capture(
        [
            nsys,
            "stats",
            "--report",
            report_name,
            "--format",
            "csv",
            "--output",
            "-",
            str(report),
        ],
        env,
    )
    output.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.context <= 0 or args.context > 4096 or args.output_tokens <= 0:
        raise ValueError("context must be 1..4096 and output-tokens must be positive")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("benchmarks/profiles") / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    cuda_environment = cuda_env.prepare_cuda_tool_environment(output_dir)
    process_env = cuda_environment.process_env
    if args.tool in {"systems", "both"}:
        nsys = require_tool("nsys")
        output_base = output_dir / "tinyllama"
        run(
            [
                nsys,
                "profile",
                "--trace=cuda,nvtx,osrt",
                "--sample=none",
                "--cpuctxsw=none",
                "--stats=false",
                f"--output={output_base}",
                *cuda_environment.wrap_target(
                    [
                        str(args.engine),
                        "benchmark-int8" if args.mode == "int8" else "benchmark",
                        str(args.model_dir),
                        str(args.context),
                        str(args.output_tokens),
                        "1",
                        "1",
                    ]
                ),
            ],
            process_env,
        )
        systems_report = output_base.with_suffix(".nsys-rep")
        require_nonempty_artifact(
            systems_report,
            "Nsight Systems",
            output_base.with_suffix(".qdstrm"),
        )
        kernel_stats = output_dir / "nsys-kernels.csv"
        capture_nsys_stats(nsys, systems_report, "cuda_gpu_kern_sum", kernel_stats, process_env)
        require_profile_rows(kernel_stats, "Nsight Systems kernel summary")
        nvtx_stats = output_dir / "nsys-nvtx.csv"
        capture_nsys_stats(nsys, systems_report, "nvtx_sum", nvtx_stats, process_env)
        require_profile_rows(
            nvtx_stats,
            "Nsight Systems NVTX summary",
            {"prefill", "decode", "sampling"},
        )
    if args.tool in {"compute", "both"}:
        ncu = require_tool("ncu")
        output_base = output_dir / "kernels"
        run(
            [
                ncu,
                "--set=full",
                "--target-processes=all",
                f"--export={output_base}",
                *cuda_environment.wrap_target(
                    [
                        str(args.kernel_benchmarks),
                        "--warmup",
                        "0",
                        "--samples",
                        "1",
                    ]
                ),
            ],
            process_env,
        )
        compute_report = output_base.with_suffix(".ncu-rep")
        require_nonempty_artifact(compute_report, "Nsight Compute")
        compute_stats = output_dir / "ncu-raw.csv"
        compute_stats.write_text(
            run_capture(
                [ncu, "--import", str(compute_report), "--page", "raw", "--csv"], process_env
            ),
            encoding="utf-8",
        )
        require_profile_rows(compute_stats, "Nsight Compute raw metrics")
    print(output_dir)


if __name__ == "__main__":
    main()
