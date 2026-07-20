from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest import mock

from scripts import compare_logits, wsl_cuda_env


class CompareLogitsTests(unittest.TestCase):
    def test_engine_subprocess_uses_cuda_environment_wrapper(self) -> None:
        environment = wsl_cuda_env.CudaToolEnvironment(
            {"LD_LIBRARY_PATH": "/wsl/driver:/usr/lib/wsl/lib"},
            ("/usr/bin/env", "LD_AUDIT=/tmp/wsl_cuda_ld_audit.so"),
            Path("/tmp/wsl_cuda_ld_audit.so"),
        )
        completed = subprocess.CompletedProcess(
            args=["engine", "logits"], returncode=0, stdout="{}\n", stderr=""
        )
        with mock.patch.object(compare_logits.subprocess, "run", return_value=completed) as run:
            result = compare_logits.run_engine(["engine", "logits"], environment)

        self.assertIs(result, completed)
        run.assert_called_once_with(
            ["/usr/bin/env", "LD_AUDIT=/tmp/wsl_cuda_ld_audit.so", "engine", "logits"],
            check=False,
            capture_output=True,
            text=True,
            env=environment.process_env,
        )


if __name__ == "__main__":
    unittest.main()
