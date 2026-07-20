from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import wsl_cuda_env


class WslCudaEnvironmentTests(unittest.TestCase):
    def test_non_wsl_preserves_target_command(self) -> None:
        with mock.patch.object(wsl_cuda_env, "is_wsl", return_value=False):
            environment = wsl_cuda_env.prepare_cuda_tool_environment(
                Path("unused"), {"PATH": "/bin"}
            )
        self.assertEqual(environment.process_env, {"PATH": "/bin"})
        self.assertEqual(environment.wrap_target(["engine", "arg"]), ["engine", "arg"])
        self.assertIsNone(environment.audit_library)

    def test_wsl_conflict_builds_target_only_audit_module(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            wsl_lib = root / "wsl-lib"
            driver_root = root / "drivers"
            driver = driver_root / "active"
            linux_libcuda = root / "linux" / "libcuda.so.1"
            output = root / "output"
            source = root / "audit.c"
            for directory in (wsl_lib, driver, linux_libcuda.parent, output):
                directory.mkdir(parents=True, exist_ok=True)
            (wsl_lib / "libcuda.so.1").write_bytes(b"wsl")
            (driver / "libnvidia-ptxjitcompiler.so.1").write_bytes(b"jit")
            linux_libcuda.write_bytes(b"linux")
            source.write_text("int value;\n", encoding="utf-8")

            def fake_run(command: list[str], **_: object) -> None:
                Path(command[-1]).write_bytes(b"audit")

            with (
                mock.patch.object(wsl_cuda_env, "is_wsl", return_value=True),
                mock.patch.object(wsl_cuda_env, "WSL_LIB_DIR", wsl_lib),
                mock.patch.object(wsl_cuda_env, "WSL_LIBCUDA", wsl_lib / "libcuda.so.1"),
                mock.patch.object(wsl_cuda_env, "WSL_DRIVER_ROOT", driver_root),
                mock.patch.object(wsl_cuda_env, "LINUX_LIBCUDA", linux_libcuda),
                mock.patch.object(wsl_cuda_env, "AUDIT_SOURCE", source),
                mock.patch.object(wsl_cuda_env.shutil, "which", return_value="/usr/bin/cc"),
                mock.patch.object(wsl_cuda_env.subprocess, "run", side_effect=fake_run) as run,
            ):
                environment = wsl_cuda_env.prepare_cuda_tool_environment(
                    output, {"LD_LIBRARY_PATH": "/existing"}
                )

            self.assertEqual(run.call_count, 1)
            self.assertIsNotNone(environment.audit_library)
            self.assertNotIn("LD_AUDIT", environment.process_env)
            wrapped = environment.wrap_target(["engine"])
            self.assertEqual(wrapped[0], "/usr/bin/env")
            self.assertTrue(wrapped[1].startswith("LD_AUDIT="))
            self.assertIn(str(driver), wrapped[2])
            self.assertEqual(wrapped[-1], "engine")

    def test_wsl_missing_mapped_driver_is_rejected(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            self.assertRaisesRegex(RuntimeError, "WSL CUDA loader is missing"),
        ):
            missing = Path(temporary_directory) / "libcuda.so.1"
            with (
                mock.patch.object(wsl_cuda_env, "is_wsl", return_value=True),
                mock.patch.object(wsl_cuda_env, "WSL_LIBCUDA", missing),
            ):
                wsl_cuda_env.prepare_cuda_tool_environment(Path(temporary_directory), {})


if __name__ == "__main__":
    unittest.main()
