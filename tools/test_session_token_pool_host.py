# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compile actual host function bodies with real headers and a FIFO mock; no SYCL.
Run: python -m unittest tools.test_session_token_pool_host -v
CXX selects a compiler; otherwise clang++/c++ on PATH. No project build needed.
Only temporary translation units/binaries are created. No models or real async I/O.
"""
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def extract(path, name):
    """Copy an entire definition, ignoring braces inside comments/string literals.

    Fail closed if the named top-level definition is absent/ambiguous. This is
    deliberately not a copied implementation that can drift from production.
    """
    text = (ROOT / path).read_text(encoding="utf-8")
    masked = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                    lambda m: " " * len(m.group()), text, flags=re.S)
    matches = list(re.finditer(
        r"^(?:\[\[noreturn\]\] )?(?:uint64_t|void|bool) " + re.escape(name) + r"\(",
        masked, flags=re.M))
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one definition of {name}")
    start = matches[0].start()
    opening = masked.index("{", matches[0].end())
    depth = 1
    end = opening + 1
    while depth:
        if masked[end] == "{":
            depth += 1
        elif masked[end] == "}":
            depth -= 1
        end += 1
    return f"// Extracted from {path}: {name}\n" + text[start:end] + "\n"


class SessionTokenPoolHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cxx = os.environ.get("CXX")
        if cxx:
            compiler = [cxx] if Path(cxx).is_file() else shlex.split(cxx)
        else:
            found = shutil.which("clang++") or shutil.which("c++")
            if not found:
                raise RuntimeError("Set CXX or put clang++/c++ on PATH")
            compiler = [found]
        cls.tmp = tempfile.TemporaryDirectory(prefix="session-token-pool-host-")
        cls.addClassCleanup(cls.tmp.cleanup)
        build = Path(cls.tmp.name)
        source = build / "extracted.cpp"
        license_header = "\n".join(
            (ROOT / "tests/core/session_token_pool_host.cpp").read_text(
                encoding="utf-8").splitlines()[:2]) + "\n"
        source.write_text(
            license_header + '#include "strata/core/session.hpp"\n'
            '#include "strata/core/device.hpp"\n'
            '#include <chrono>\n#include <cstring>\n#include <exception>\n#include <limits>\n'
            'namespace strata::core {\n' +
            "".join(extract("src/core/session.cpp", n) for n in
                    ("gdn_state_floats", "gdn_point_at", "session_token_pool")) +
            '}\nnamespace strata::kernels {\nnamespace {\n' +
            extract("src/kernels/xe/qsa.cpp", "fail") + '}\n' +
            extract("src/kernels/xe/qsa.cpp", "qsa_step_fill") + '}\n',
            encoding="utf-8")
        cls.exe = build / ("host.exe" if os.name == "nt" else "host")
        command = compiler + ["-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror",
                              "-I", str(ROOT / "include"), str(source),
                              str(ROOT / "tests/core/session_token_pool_host.cpp"),
                              "-o", str(cls.exe)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(f"compiler: {compiler}\n{result.stdout}{result.stderr}")
        print(f"host compiler: {compiler[0]}", flush=True)
        if result.stdout or result.stderr:
            print(result.stdout + result.stderr, flush=True)

    def run_case(self, case):
        result = subprocess.run([str(self.exe), case], capture_output=True,
                                text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS " + case, result.stdout)
        print(result.stdout.strip(), flush=True)

    def test_orchestration(self):
        self.run_case("orchestration")

    def test_guards(self):
        self.run_case("guards")

    def test_failures(self):
        self.run_case("failures")

    def test_qsa_step_arithmetic(self):
        self.run_case("qsa-step")

    def test_layer_dump(self):
        self.run_case("layer-dump")

    def test_host_timing(self):
        self.run_case("timing")


if __name__ == "__main__":
    unittest.main()
