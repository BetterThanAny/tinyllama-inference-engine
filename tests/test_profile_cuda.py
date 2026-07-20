from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import profile_cuda, run_cuda_memcheck


class ProfileArtifactTests(unittest.TestCase):
    def test_accepts_nonempty_importable_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = Path(temporary_directory) / "tinyllama.nsys-rep"
            report.write_bytes(b"report")

            profile_cuda.require_nonempty_artifact(report, "Nsight Systems")

    def test_rejects_raw_capture_without_importable_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report = root / "tinyllama.nsys-rep"
            raw_capture = root / "tinyllama.qdstrm"
            raw_capture.write_bytes(b"raw capture")

            with self.assertRaisesRegex(RuntimeError, "raw capture exists"):
                profile_cuda.require_nonempty_artifact(report, "Nsight Systems", raw_capture)

    def test_accepts_nonempty_kernel_rows(self) -> None:
        content = 'Time (%),Instances,Name\n100.0,1,"RmsNormKernel(float*)"\n'
        with tempfile.TemporaryDirectory() as temporary_directory:
            stats = Path(temporary_directory) / "kernels.csv"
            stats.write_text(content, encoding="utf-8")
            profile_cuda.require_profile_rows(stats, "Nsight Systems kernel summary")

    def test_rejects_zero_profile_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            stats = Path(temporary_directory) / "kernels.csv"
            stats.write_text("Time (%),Instances,Name\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "0 data rows"):
                profile_cuda.require_profile_rows(stats, "Nsight Systems kernel summary")

    def test_requires_all_end_to_end_nvtx_ranges(self) -> None:
        content = "Time (%),Instances,Range\n50.0,1,prefill\n50.0,1,decode\n"
        with tempfile.TemporaryDirectory() as temporary_directory:
            stats = Path(temporary_directory) / "nvtx.csv"
            stats.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "sampling"):
                profile_cuda.require_profile_rows(
                    stats,
                    "Nsight Systems NVTX summary",
                    {"prefill", "decode", "sampling"},
                )

    def test_memcheck_requires_explicit_clean_summary(self) -> None:
        clean = (
            "========= LEAK SUMMARY: 0 bytes leaked in 0 allocations\n"
            "========= ERROR SUMMARY: 0 errors\n"
        )
        run_cuda_memcheck.require_clean_sanitizer(clean, "memcheck")
        with self.assertRaisesRegex(RuntimeError, "did not instrument"):
            run_cuda_memcheck.require_clean_sanitizer(
                "Target application terminated before first instrumented API call", "memcheck"
            )
        with self.assertRaisesRegex(RuntimeError, "0 errors"):
            run_cuda_memcheck.require_clean_sanitizer(
                "========= LEAK SUMMARY: 0 bytes leaked in 0 allocations\n"
                "========= ERROR SUMMARY: 1 error\n",
                "memcheck",
            )
        with self.assertRaisesRegex(RuntimeError, "zero-leak"):
            run_cuda_memcheck.require_clean_sanitizer(
                "========= ERROR SUMMARY: 0 errors\n", "memcheck"
            )

    def test_racecheck_requires_explicit_clean_summary(self) -> None:
        run_cuda_memcheck.require_clean_sanitizer(
            "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n",
            "racecheck",
        )
        with self.assertRaisesRegex(RuntimeError, "clean RACECHECK"):
            run_cuda_memcheck.require_clean_sanitizer(
                "========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)\n",
                "racecheck",
            )

    def test_memcheck_target_arguments_accept_standard_separator(self) -> None:
        self.assertEqual(run_cuda_memcheck.normalize_target_args(["--", "--test"]), ["--test"])

    def test_memcheck_command_accepts_packaged_injection_path(self) -> None:
        environment = run_cuda_memcheck.cuda_env.CudaToolEnvironment({}, (), None)
        command = run_cuda_memcheck.build_command(
            "/usr/bin/compute-sanitizer",
            environment,
            Path("target"),
            ["--test"],
            Path("/tool/injection"),
        )
        self.assertEqual(command[1:3], ["--injection-path", "/tool/injection"])

    def test_racecheck_command_does_not_request_memcheck_leaks(self) -> None:
        environment = run_cuda_memcheck.cuda_env.CudaToolEnvironment({}, (), None)
        command = run_cuda_memcheck.build_command(
            "compute-sanitizer", environment, Path("target"), [], tool="racecheck"
        )
        self.assertIn("racecheck", command)
        self.assertNotIn("--leak-check", command)


if __name__ == "__main__":
    unittest.main()
