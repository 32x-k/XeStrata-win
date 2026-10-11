# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Small production scalar-decoding checks; no model, GPU, or upstream executable."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DequantHostContract(unittest.TestCase):
    def test_scalar_contracts(self):
        cxx = os.environ.get("CXX")
        if cxx:
            compiler = [cxx] if Path(cxx).is_file() else shlex.split(cxx)
        else:
            found = shutil.which("clang++") or shutil.which("c++")
            if not found:
                raise RuntimeError("Set CXX or put clang++/c++ on PATH")
            compiler = [found]
        with tempfile.TemporaryDirectory(prefix="dequant-host-contract-") as temp:
            exe = Path(temp) / ("check.exe" if os.name == "nt" else "check")
            result = subprocess.run(compiler + ["-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror",
                                    "-D_CRT_SECURE_NO_WARNINGS", "-I", str(ROOT / "include"),
                                    str(ROOT / "tests/core/dequant_host_contract.cpp"), "-o", str(exe)],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS:", result.stdout)
            print(result.stdout.strip(), flush=True)


if __name__ == "__main__":
    unittest.main()
