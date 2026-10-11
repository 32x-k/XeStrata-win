# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""CPU-only wrapper tests: local Git fixtures, real git apply, mocked CMake.

    python -m unittest tools.test_adaptivecpp_build
"""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import adaptivecpp_build as wrapper

REAL_COMMIT = wrapper.COMMIT
REAL_ORIGINAL_SHA256 = wrapper.ORIGINAL_SHA256
REAL_ORIGINAL_CRLF_SHA256 = wrapper.ORIGINAL_CRLF_SHA256
COPYRIGHT = b"// Copyright The AdaptiveCpp Contributors. All rights reserved.\n// BSD-2-Clause notice kept.\n"
DEFAULT_EVENT = b"  event() {}\n"
CONTEXT = (b"      const rt::dag_node_ptr &evt,\n"
           b"      async_handler handler =\n"
           b"          [](exception_list e) { glue::default_async_handler(e); })\n")
TAIL = b"\n  std::vector<event> get_wait_list()\n  {\n"
# The hunk starts at line 44, as in the real pinned patch. No upstream download
# is needed: only the fixture commit/hash constants are substituted in tests.
ORIGINAL = COPYRIGHT + DEFAULT_EVENT + b"\n" * 40 + CONTEXT + wrapper.OLD + TAIL
EXPECTED = ORIGINAL.replace(wrapper.OLD, wrapper.NEW, 1)
PATCH_BYTES = (b"# Local test patch\n\n" +
               f"diff --git a/{wrapper.HEADER} b/{wrapper.HEADER}\n"
               f"--- a/{wrapper.HEADER}\n+++ b/{wrapper.HEADER}\n"
               "@@ -44,7 +44,7 @@ public:\n".encode() +
               b"".join(b" " + line for line in CONTEXT.splitlines(keepends=True)) +
               b"-" + wrapper.OLD + b"+" + wrapper.NEW +
               b"".join(b" " + line for line in TAIL.splitlines(keepends=True)))


class AdaptiveCppBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        self.header = self.source / wrapper.HEADER
        self.header.parent.mkdir(parents=True)
        self.header.write_bytes(ORIGINAL)
        self.other = self.source / "user-file.txt"
        self.other.write_bytes(b"committed\n")
        self.fixture_git("init", "--quiet")
        self.fixture_git("config", "core.autocrlf", "false")
        self.fixture_git("add", ".")
        self.fixture_git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                         "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture original header")
        self.commit = self.fixture_git("rev-parse", "HEAD").decode().strip()
        self.patch = self.root / "event.patch"
        self.patch.write_bytes(PATCH_BYTES)
        self.prefix = self.root / "install"
        self.build = self.root / "build"
        self.installed = self.prefix / wrapper.INSTALLED_HEADER
        self.receipt = self.root / "receipt.json"
        for name, value in (("COMMIT", self.commit), ("ORIGINAL_SHA256", wrapper.sha256(ORIGINAL)),
                            ("ORIGINAL_CRLF_SHA256", wrapper.sha256(ORIGINAL.replace(b"\n", b"\r\n"))),
                            ("PATCH", self.patch)):
            patcher = mock.patch.object(wrapper, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.other.write_bytes(b"user modifications must survive\n")

    def fixture_git(self, *args):
        return subprocess.run(["git", "-C", str(self.source), *args], check=True,
                              capture_output=True).stdout

    def install_fixture(self, content=EXPECTED):
        self.installed.parent.mkdir(parents=True, exist_ok=True)
        self.installed.write_bytes(content)

    def cli(self, *args):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = wrapper.main(["--source", str(self.source), *args])
        return code, output.getvalue(), errors.getvalue()

    def snapshot(self):
        return {str(path.relative_to(self.root)): (path.read_bytes(), path.stat().st_mtime_ns)
                for path in self.root.rglob("*") if path.is_file() and ".git" not in path.parts}

    def test_real_pins_and_parent_patch_body(self):
        # Production constants are tested independently of fixture substitutions.
        self.assertEqual(REAL_COMMIT, "0ea863abe1af4422593583f91e5608ef75044a16")
        self.assertEqual(REAL_ORIGINAL_SHA256, "0576c693880346dcd6d85ec3934be332c8e1678af7f126c267d603916338b2ad")
        self.assertEqual(REAL_ORIGINAL_CRLF_SHA256, "0eb54a547e68eaf0055f75431fbc8af7186e7221988d0829c333acec76797526")
        with mock.patch.object(wrapper, "COMMIT", REAL_COMMIT), \
                mock.patch.object(wrapper, "ORIGINAL_SHA256", REAL_ORIGINAL_SHA256):
            with self.assertRaisesRegex(wrapper.BuildError, "HEAD must be exactly"):
                wrapper.check_source(self.source)
        parent_patch = Path(wrapper.__file__).resolve().parents[1] / \
            "third_party/main/adaptivecpp/patches/0001-event-async-handler.patch"
        wrapper.check_patch_body(parent_patch.read_bytes())
        checked = wrapper.check_source(self.source, parent_patch)
        self.assertEqual(checked.expected, EXPECTED)

    def test_nonempty_unknown_prefix_stops_before_patch(self):
        self.prefix.mkdir()
        (self.prefix / "existing-user-file").write_bytes(b"must survive")
        before = self.snapshot()
        code, _, _ = self.cli("--build", str(self.build), "--prefix", str(self.prefix))
        self.assertEqual(code, 1)
        self.assertEqual(self.snapshot(), before)

    def test_nonempty_unpatched_prefix_stops_before_patch(self):
        self.install_fixture(ORIGINAL)
        before = self.snapshot()
        code, _, error = self.cli("--build", str(self.build), "--prefix", str(self.prefix))
        self.assertEqual(code, 1)
        self.assertIn("installed header", error)
        self.assertEqual(self.snapshot(), before)

    def test_apply_only_real_git_and_receipt(self):
        with mock.patch.object(wrapper, "run", wraps=wrapper.run) as calls:
            code, output, errors = self.cli("--apply-only", "--receipt", str(self.receipt))
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual(self.header.read_bytes(), EXPECTED)
        self.assertTrue(self.header.read_bytes().startswith(COPYRIGHT))
        self.assertIn(DEFAULT_EVENT, self.header.read_bytes())
        self.assertEqual(self.other.read_bytes(), b"user modifications must survive\n")
        self.assertFalse(self.build.exists())
        commands = [call.args[0] for call in calls.call_args_list]
        self.assertTrue(all(command[0] == "git" for command in commands))
        check_index = next(i for i, cmd in enumerate(commands) if "--check" in cmd)
        apply_index = next(i for i, cmd in enumerate(commands) if "apply" in cmd and "--check" not in cmd)
        self.assertLess(check_index, apply_index)
        receipt = json.loads(output)
        self.assertEqual(json.loads(self.receipt.read_text()), receipt)
        self.assertEqual(receipt["adaptivecpp_commit"], self.commit)
        self.assertEqual(receipt["patch_disposition"], "applied")
        self.assertEqual(receipt["patch_sha256"], wrapper.sha256(PATCH_BYTES))
        self.assertEqual(receipt["source_header_sha256"], wrapper.sha256(EXPECTED))
        self.assertIsNone(receipt["installed_header_sha256"])

    def test_already_patched_skips_apply_without_changes(self):
        self.header.write_bytes(EXPECTED)
        before = self.snapshot()
        with mock.patch.object(wrapper, "run", wraps=wrapper.run) as calls:
            code, output, errors = self.cli("--apply-only")
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual(json.loads(output)["patch_disposition"], "already-applied-skip")
        self.assertEqual(self.snapshot(), before)
        apply_commands = [call.args[0] for call in calls.call_args_list if "apply" in call.args[0]]
        self.assertEqual(len(apply_commands), 1)
        self.assertIn("--check", apply_commands[0])
        self.assertIn("--reverse", apply_commands[0])

    def test_wrong_commit_stops_without_mutation(self):
        before = self.snapshot()
        with mock.patch.object(wrapper, "COMMIT", "a" * 40):
            code, _, errors = self.cli("--apply-only")
        self.assertEqual(code, 1)
        self.assertIn("HEAD must be exactly", errors)
        self.assertEqual(self.snapshot(), before)

    def test_wrong_committed_hash_stops(self):
        before = self.snapshot()
        with mock.patch.object(wrapper, "ORIGINAL_SHA256", "0" * 64):
            code, _, errors = self.cli("--apply-only")
        self.assertEqual(code, 1)
        self.assertIn("pinned original SHA256", errors)
        self.assertEqual(self.snapshot(), before)

    def test_user_header_changes_are_never_overwritten(self):
        for content in (ORIGINAL + b"// user edit\n", EXPECTED + b"// user edit\n",
                        ORIGINAL.replace(b"event() {}", b"event() { user(); }")):
            with self.subTest(content=content[-30:]):
                self.header.write_bytes(content)
                before = self.snapshot()
                code, _, errors = self.cli("--apply-only")
                self.assertEqual(code, 1)
                self.assertIn("neither the exact original", errors)
                self.assertEqual(self.snapshot(), before)

    def test_bad_patch_body_stops_even_when_already_patched(self):
        bad_patches = (PATCH_BYTES.replace(wrapper.NEW, b"      : _node{evt}, _handler{std::move(handler)} {}\n"),
                       PATCH_BYTES + b"+// extra change\n", PATCH_BYTES + PATCH_BYTES,
                       PATCH_BYTES.replace(b"b/include/hipSYCL", b"b/other"))
        for header in (ORIGINAL, EXPECTED):
            for bad_patch in bad_patches:
                with self.subTest(header=header[-30:], patch=bad_patch[-30:]):
                    self.header.write_bytes(header)
                    self.patch.write_bytes(bad_patch)
                    before = self.snapshot()
                    code, _, _ = self.cli("--apply-only")
                    self.assertEqual(code, 1)
                    self.assertEqual(self.snapshot(), before)

    def test_bad_patch_context_fails_git_check_without_changes(self):
        self.patch.write_bytes(PATCH_BYTES.replace(b"glue::default_async_handler", b"glue::wrong_handler"))
        for header in (ORIGINAL, EXPECTED):
            with self.subTest(patched=header == EXPECTED):
                self.header.write_bytes(header)
                before = self.snapshot()
                code, _, errors = self.cli("--apply-only")
                self.assertEqual(code, 1)
                self.assertIn("git failed", errors)
                self.assertEqual(self.snapshot(), before)

    def test_header_changed_after_check_stops(self):
        checked = wrapper.check_source(self.source)
        self.header.write_bytes(ORIGINAL + b"// concurrent edit\n")
        before = self.snapshot()
        with self.assertRaisesRegex(wrapper.BuildError, "changed after validation"):
            wrapper.apply_patch(checked)
        self.assertEqual(self.snapshot(), before)

    def test_verify_only_never_mutates_even_original_source(self):
        self.install_fixture()
        before = self.snapshot()
        with mock.patch.object(wrapper, "apply_patch", side_effect=AssertionError("must not apply")):
            code, output, errors = self.cli("--verify-installed-only", "--prefix", str(self.prefix))
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual(self.snapshot(), before)
        receipt = json.loads(output)
        self.assertEqual(receipt["patch_disposition"], "verification-only")
        self.assertEqual(receipt["installed_header_sha256"], wrapper.sha256(EXPECTED))
        self.assertEqual(receipt["source_header_sha256"], wrapper.sha256(ORIGINAL))
        self.assertFalse(self.receipt.exists())

    def test_installed_missing_original_or_changed_is_rejected(self):
        for content in (None, ORIGINAL, EXPECTED + b"// modified install\n"):
            with self.subTest(content=content):
                if content is not None:
                    self.install_fixture(content)
                before = self.snapshot()
                code, _, _ = self.cli("--verify-installed-only", "--prefix", str(self.prefix))
                self.assertEqual(code, 1)
                self.assertEqual(self.snapshot(), before)

    def test_path_and_cache_checks_happen_before_apply(self):
        paths = ((self.source, self.prefix), (self.source / "out", self.prefix),
                 (self.build, self.source), (self.build, self.source / "install"),
                 (self.root, self.prefix), (self.build, self.build / "install"))
        for build, prefix in paths:
            with self.subTest(build=build, prefix=prefix):
                before = self.snapshot()
                code, _, _ = self.cli("--build", str(build), "--prefix", str(prefix))
                self.assertEqual(code, 1)
                self.assertEqual(self.snapshot(), before)
        self.build.mkdir()
        for key in ("CMAKE_HOME_DIRECTORY", "CMAKE_INSTALL_PREFIX"):
            with self.subTest(key=key):
                (self.build / "CMakeCache.txt").write_text(f"{key}:PATH={self.root / 'other'}\n")
                before = self.snapshot()
                code, _, errors = self.cli("--build", str(self.build), "--prefix", str(self.prefix))
                self.assertEqual(code, 1)
                self.assertIn("existing CMake cache", errors)
                self.assertEqual(self.snapshot(), before)

    def test_receipt_not_overwritten_and_not_inside_source(self):
        self.receipt.write_text("user receipt")
        for receipt in (self.receipt, self.source / "new.json"):
            with self.subTest(receipt=receipt):
                before = self.snapshot()
                code, _, _ = self.cli("--apply-only", "--receipt", str(receipt))
                self.assertEqual(code, 1)
                self.assertEqual(self.snapshot(), before)

    def test_cmake_path_mode_overrides_rejected_before_apply(self):
        for argument in ("-Sother", "-Bother", "--install-prefix=/other", "--preset=other",
                         "-DCMAKE_INSTALL_PREFIX:PATH=/other", "-Pscript.cmake", "--build=other"):
            with self.subTest(argument=argument):
                before = self.snapshot()
                code, _, _ = self.cli("--build", str(self.build), "--prefix", str(self.prefix),
                                      f"--cmake-arg={argument}")
                self.assertEqual(code, 1)
                self.assertEqual(self.snapshot(), before)

    def test_normal_flow_mocked_cmake_apply_before_configure(self):
        commands = []
        real_run = wrapper.run

        def fake_run(args, **kwargs):
            if args[0] != "cmake":
                return real_run(args, **kwargs)
            self.assertEqual(self.header.read_bytes(), EXPECTED)
            self.assertTrue(kwargs["verbose"])
            commands.append(args)
            if "--install" in args:
                self.install_fixture()
            return b""

        with mock.patch.object(wrapper, "run", side_effect=fake_run):
            code, output, errors = self.cli("--build", str(self.build), "--prefix", str(self.prefix),
                                            "--cmake-arg=-G", "--cmake-arg=Ninja",
                                            "--cmake-arg=-DACPP_COMPILER_FEATURE_PROFILE=none",
                                            "--jobs", "2", "--receipt", str(self.receipt))
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual(commands, [
            ["cmake", "-G", "Ninja", "-DACPP_COMPILER_FEATURE_PROFILE=none", "-S", str(self.source),
             "-B", str(self.build), f"-DCMAKE_INSTALL_PREFIX={self.prefix}"],
            ["cmake", "--build", str(self.build), "--parallel", "2"],
            ["cmake", "--install", str(self.build)]])
        self.assertEqual(json.loads(output)["installed_header_sha256"], wrapper.sha256(EXPECTED))
        self.assertEqual(json.loads(self.receipt.read_text()), json.loads(output))

    def test_failed_cmake_does_not_revert_patch_or_write_receipt(self):
        real_run = wrapper.run

        def fake_run(args, **kwargs):
            if args[0] == "cmake":
                raise wrapper.BuildError("mock configure failed")
            return real_run(args, **kwargs)

        with mock.patch.object(wrapper, "run", side_effect=fake_run):
            code, _, errors = self.cli("--build", str(self.build), "--prefix", str(self.prefix),
                                       "--receipt", str(self.receipt))
        self.assertEqual(code, 1)
        self.assertIn("mock configure failed", errors)
        self.assertEqual(self.header.read_bytes(), EXPECTED)
        self.assertFalse(self.receipt.exists())

    def test_jobs_positive_and_default(self):
        self.assertEqual(wrapper.positive_jobs("2"), 2)
        real_run = wrapper.run
        commands = []

        def fake_run(args, **kwargs):
            if args[0] != "cmake":
                return real_run(args, **kwargs)
            commands.append(args)
            if "--install" in args:
                self.install_fixture()
            return b""

        with mock.patch.object(wrapper, "run", side_effect=fake_run):
            code, _, errors = self.cli("--build", str(self.build), "--prefix", str(self.prefix))
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual(commands[1][-2:], ["--parallel", "2"])
        for jobs in ("0", "-1", "bad", "1.5"):
            with self.subTest(jobs=jobs), contextlib.redirect_stderr(io.StringIO()):
                before = self.snapshot()
                with self.assertRaises(SystemExit):
                    wrapper.main(["--source", str(self.source), "--apply-only", "--jobs", jobs])
                self.assertEqual(self.snapshot(), before)

    def test_exact_lf_crlf_variants_apply_skip_and_verify(self):
        for ending in ("LF", "CRLF"):
            original = ORIGINAL if ending == "LF" else ORIGINAL.replace(b"\n", b"\r\n")
            expected = EXPECTED if ending == "LF" else EXPECTED.replace(b"\n", b"\r\n")
            for patched in (False, True):
                with self.subTest(ending=ending, patched=patched):
                    self.header.write_bytes(expected if patched else original)
                    before_mtime = self.header.stat().st_mtime_ns
                    with mock.patch.object(wrapper, "git", wraps=wrapper.git) as calls:
                        code, output, errors = self.cli("--apply-only")
                    self.assertEqual((code, errors), (0, ""))
                    self.assertEqual(self.header.read_bytes(), expected)
                    if patched:
                        self.assertEqual(self.header.stat().st_mtime_ns, before_mtime)
                    data = PATCH_BYTES if ending == "LF" else PATCH_BYTES.replace(b"\n", b"\r\n")
                    for call in calls.call_args_list:
                        if "apply" in call.args:
                            self.assertEqual(call.kwargs["data"], data)
                    receipt = json.loads(output)
                    self.assertEqual(receipt["patch_sha256"], wrapper.sha256(PATCH_BYTES))
                    self.assertEqual(receipt["source_header_sha256"], wrapper.sha256(expected))
                    self.assertEqual(receipt["original_header_sha256"], wrapper.sha256(original))
                    self.assertEqual(receipt["committed_header_sha256"], wrapper.sha256(ORIGINAL))
                    self.assertEqual(receipt["source_line_ending"], ending)
                    self.install_fixture(expected)
                    before = self.snapshot()
                    code, output, errors = self.cli("--verify-installed-only", "--prefix", str(self.prefix))
                    self.assertEqual((code, errors), (0, ""))
                    self.assertEqual(json.loads(output)["installed_header_sha256"], wrapper.sha256(expected))
                    self.assertEqual(self.snapshot(), before)

    def test_mixed_or_other_line_endings_rejected_without_changes(self):
        for content in (ORIGINAL, EXPECTED):
            for changed in (content.replace(b"\n", b"\r\n", 1),
                            content.replace(b"\n", b"\r\n").replace(b"\r\n", b"\n", 1),
                            content.replace(b"\n", b"\r")):
                with self.subTest(changed=changed[:100]):
                    self.header.write_bytes(changed)
                    before = self.snapshot()
                    code, _, errors = self.cli("--apply-only")
                    self.assertEqual(code, 1)
                    self.assertIn("neither the exact original", errors)
                    self.assertEqual(self.snapshot(), before)

    def test_crlf_patch_file_hashes_canonical_lf_bytes(self):
        self.patch.write_bytes(PATCH_BYTES.replace(b"\n", b"\r\n"))
        self.header.write_bytes(ORIGINAL.replace(b"\n", b"\r\n"))
        code, output, errors = self.cli("--apply-only")
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual(json.loads(output)["patch_sha256"], wrapper.sha256(PATCH_BYTES))
        self.assertEqual(self.header.read_bytes(), EXPECTED.replace(b"\n", b"\r\n"))

    def test_command_failure_reports_stdout_and_stderr(self):
        result = subprocess.CompletedProcess(["cmake"], 1, b"Ninja failure on stdout\n", b"compiler warning\n")
        with mock.patch.object(wrapper.subprocess, "run", return_value=result):
            with self.assertRaises(wrapper.BuildError) as error:
                wrapper.run(["cmake", "--build", "unused"])
        self.assertIn("Ninja failure on stdout", str(error.exception))
        self.assertIn("compiler warning", str(error.exception))

    def test_verbose_command_output_goes_to_stderr_not_json_stdout(self):
        result = subprocess.CompletedProcess(["cmake"], 0, b"build progress\n", b"compiler warning\n")
        output, log = io.StringIO(), io.StringIO()
        with mock.patch.object(wrapper.subprocess, "run", return_value=result), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(log):
            wrapper.run(["cmake", "--build", "unused"], verbose=True)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("+ cmake --build unused", log.getvalue())
        self.assertIn("build progress", log.getvalue())
        self.assertIn("compiler warning", log.getvalue())

    def test_utf8_cmake_cache(self):
        self.build.mkdir()
        (self.build / "CMakeCache.txt").write_text(
            f"// コンパイラ\nCMAKE_HOME_DIRECTORY:INTERNAL={self.source}\n"
            f"CMAKE_INSTALL_PREFIX:PATH={self.prefix}\n", encoding="utf-8")
        wrapper.check_paths(self.source, self.build, self.prefix, None)

    def test_verify_mode_refuses_receipt_without_mutation(self):
        self.install_fixture()
        before = self.snapshot()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            wrapper.main(["--source", str(self.source), "--prefix", str(self.prefix),
                          "--verify-installed-only", "--receipt", str(self.receipt)])
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
